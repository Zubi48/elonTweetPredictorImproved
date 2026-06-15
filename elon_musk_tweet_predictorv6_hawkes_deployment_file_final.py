"""
================================================================================
 Elon Musk Weekly Tweet Count — Bayesian Predictor w/ News Event Factor
 + Hawkes self-exciting point process (MLE-fitted, exponential kernel)
   for burst amplification and post-burst decay of the forecast
 + Conditional-probability engine: latent-regime (hyperexponential) gap
   model for silence conditioning, diurnal/weekly exposure weighting,
   and conjugate Gamma–Poisson window calibration
 v6: Hawkes process fitted & evaluated on the activity-rescaled
   ("effective") time axis, so bursts during normally quiet hours generate
   STRONGER excitation instead of being diluted by the flat diurnal
   multiplier; live event arrays refreshed on every forecast call so the
   excitation no longer goes stale (and decays away unexpressed) between
   throttled 30-minute MLE refits; window-calibration multiplier clamp
   widened from [0.5, 2.0] to [0.4, 3.0] so hot weeks are tracked
 REST polling every 5 minutes via XTracker API
 Discord webhook notification on every new tweet detected
================================================================================
"""

import asyncio
import os
import aiohttp
import csv
import logging
import pickle
import signal
import sys
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from scipy.optimize import minimize
from scipy.stats import norm

import numpy as np
import pandas as pd
import pytz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

XTRACKER_BASE_URL     = "https://xtracker.polymarket.com/api"
ELON_HANDLE           = "elonmusk"
PLATFORM              = "X"

_DATA_DIR             = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent))
CSV_FILE              = str(_DATA_DIR / "elonmusk_tweet_history_improved.csv")
MODEL_FILE            = str(_DATA_DIR / "bayesian_model_improved.pkl")
LOG_FILE              = str(_DATA_DIR / "tweet_predictor_improved.log")
EVENT_LOG_FILE        = str(_DATA_DIR / "event_factors_improved.log")

NEWS_API_KEY          = "INSERT HERE"
NEWS_API_BASE_URL     = "https://newsapi.org/v2/everything"
NEWS_SCAN_INTERVAL    = 1800
NEWS_LOOKBACK_HOURS   = 12
MAX_ARTICLES_PER_SCAN = 30

DISCORD_WEBHOOK_URL   = "https://discord.com/api/webhooks/1514384038134288464/Y4kW18zp6ImRBXTXc4_tisfXqUXchK7vyk6el0bOic6I-DMecBmMQ3iwazRAp6iRL-xK"

DEVIATION_Z_THRESHOLD = 1.75
POLL_INTERVAL_SEC     = 300
BET_CHECK_INTERVAL    = 300
EVENT_FACTOR_DECAY    = 0.90
EVENT_FACTOR_MAX      = 0.40

# ── Hawkes / conditional-forecast engine ──
HAWKES_REFIT_INTERVAL  = 1800      # seconds between MLE refits (throttle)
HAWKES_MIN_EVENTS      = 50        # minimum tweet events required to fit
HAWKES_MAX_FIT_EVENTS  = 4000      # most-recent events used in the MLE
HAWKES_MAX_BRANCHING   = 0.99      # subcriticality cap on n = α/β
HAWKES_FANO_MAX        = 25.0      # cap on stationary Fano factor 1/(1−n)²
SILENCE_FACTOR_MIN     = 0.25      # floor on hazard-based silence multiplier
SILENCE_FACTOR_MAX     = 1.75      # cap on the same multiplier
WINDOW_CALIB_PRIOR     = 10.0      # Gamma(a0, a0) prior strength (pseudo-tweets)
FORECAST_HORIZON_CAP_H = 35 * 24.0 # numerical-integration horizon cap (hours)
EVENT_REFRESH_MAX_AGE_S = 60.0     # max staleness of live event arrays (sec)
THETA_MIN              = 0.4       # window-calibration multiplier clamp (lo)
THETA_MAX              = 3.0       # window-calibration multiplier clamp (hi)

EST_TZ                = pytz.timezone("America/New_York")

# ──────────────────────────────────────────────────────────────────────────────
# BET ANSWER INTERVALS
# ──────────────────────────────────────────────────────────────────────────────
ALL_INTERVALS: list[tuple[str, int, int]] = [
    ("<20",      0,   19),
    ("20-39",   20,   39),
    ("40-59",   40,   59),
    ("60-79",   60,   79),
    ("80-99",   80,   99),
    ("100-119", 100, 119),
    ("120-139", 120, 139),
    ("140-159", 140, 159),
    ("160-179", 160, 179),
    ("180-199", 180, 199),
    ("200-219", 200, 219),
    ("220-239", 220, 239),
    ("240-259", 240, 259),
    ("260-279", 260, 279),
    ("280-299", 280, 299),
    ("300-319", 300, 319),
    ("320-339", 320, 339),
    ("340-359", 340, 359),
    ("360-379", 360, 379),
    ("380-399", 380, 399),
    ("400-419", 400, 419),
    ("420-439", 420, 439),
    ("440-459", 440, 459),
    ("460-479", 460, 479),
    ("480-499", 480, 499),
    ("500-519", 500, 519),
    ("520-539", 520, 539),
    ("540-559", 540, 559),
    ("560-579", 560, 579),
    ("580-599", 580, 599),
    ("600-619", 600, 619),
    ("620-639", 620, 639),
    ("640-659", 640, 659),
    ("660-679", 660, 679),
    ("680-699", 680, 699),
]

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("TweetPredictor")

event_logger = logging.getLogger("EventFactor")
event_logger.setLevel(logging.INFO)
_efh = logging.FileHandler(EVENT_LOG_FILE, encoding="utf-8", mode="w")
_efh.setFormatter(logging.Formatter("%(asctime)s — %(message)s"))
event_logger.addHandler(_efh)
event_logger.propagate = True

# ──────────────────────────────────────────────────────────────────────────────
# NEWS KEYWORD TAXONOMY
# ──────────────────────────────────────────────────────────────────────────────

EVENT_TAXONOMY = {
    "geopolitical": {
        "queries":   ["war", "sanctions", "NATO", "China Taiwan", "Middle East",
                      "Russia Ukraine", "election", "coup", "treaty"],
        "keywords":  ["war", "conflict", "sanction", "nato", "military",
                      "invasion", "ceasefire", "election", "president",
                      "diplomacy", "nuclear", "ally", "taiwan", "ukraine",
                      "russia", "israel", "iran", "coup", "regime"],
        "weight":    0.35,
        "direction": +1,
    },
    "economic": {
        "queries":   ["Federal Reserve interest rate", "inflation CPI",
                      "stock market crash", "recession", "cryptocurrency bitcoin",
                      "tariff trade war", "GDP", "bank collapse"],
        "keywords":  ["fed", "inflation", "rate hike", "recession", "gdp",
                      "market crash", "bitcoin", "crypto", "tariff",
                      "unemployment", "cpi", "interest rate", "bank",
                      "treasury", "s&p", "nasdaq", "dow"],
        "weight":    0.30,
        "direction": +1,
    },
    "tesla": {
        "queries":   ["Tesla stock", "Tesla recall", "Tesla earnings",
                      "Tesla Cybertruck", "Tesla Autopilot", "Tesla layoffs",
                      "TSLA"],
        "keywords":  ["tesla", "tsla", "cybertruck", "autopilot", "fsd",
                      "recall", "layoff", "earnings", "delivery", "gigafactory",
                      "model 3", "model y", "model s", "roadster", "semi"],
        "weight":    0.25,
        "direction": +1,
    },
    "spacex_xai": {
        "queries":   ["SpaceX launch", "Starship", "xAI Grok", "Neuralink",
                      "Boring Company"],
        "keywords":  ["spacex", "starship", "falcon", "rocket", "launch",
                      "xai", "grok", "neuralink", "boring company", "starlink"],
        "weight":    0.10,
        "direction": +1,
    },
    "personal_legal": {
        "queries":   ["Elon Musk lawsuit", "Elon Musk controversy",
                      "Elon Musk SEC", "Elon Musk court"],
        "keywords":  ["lawsuit", "sued", "court", "sec", "investigation",
                      "controversy", "scandal", "fired", "arrested"],
        "weight":    0.10,
        "direction": +1,
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# CSV MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class CSVManager:
    HEADERS = ["DateTime_UTC", "Cumulative_Tweet_Count"]

    def __init__(self, filepath: str = CSV_FILE):
        self.filepath = Path(filepath)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.HEADERS).writeheader()
            logger.info("Created CSV: %s", self.filepath)

    def append_row(self, datetime_utc: datetime, cumulative_count: int) -> None:
        row = {
            "DateTime_UTC":           datetime_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Cumulative_Tweet_Count": cumulative_count,
        }
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.HEADERS).writerow(row)
        logger.debug("CSV row written: %s", row)

    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath, parse_dates=["DateTime_UTC"])
        df["DateTime_UTC"] = pd.to_datetime(df["DateTime_UTC"], utc=True)
        df.sort_values("DateTime_UTC", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def latest_cumulative_count(self) -> int:
        df = self.load_dataframe()
        return 0 if df.empty else int(df["Cumulative_Tweet_Count"].iloc[-1])

    def latest_timestamp(self) -> Optional[datetime]:
        df = self.load_dataframe()
        return None if df.empty else df["DateTime_UTC"].iloc[-1].to_pydatetime()


# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL PATTERN ANALYZER
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPatternAnalyzer:
    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

    def __init__(self, csv_manager: CSVManager):
        self.csv = csv_manager

    def analyze_and_log(self) -> dict:
        df = self.csv.load_dataframe()
        if df.empty or len(df) < 2:
            logger.warning("TemporalPatternAnalyzer: not enough data yet.")
            return {i: 1 / 7 for i in range(7)}

        df_est = df.copy()
        df_est["DateTime_EST"] = df_est["DateTime_UTC"].dt.tz_convert(EST_TZ)
        df_est["DayOfWeek"]    = df_est["DateTime_EST"].dt.dayofweek
        df_est["Hour"]         = df_est["DateTime_EST"].dt.hour
        df_est["DailyCount"]   = (
            df_est["Cumulative_Tweet_Count"].diff()
            .fillna(df_est["Cumulative_Tweet_Count"].iloc[0])
            .clip(lower=0).astype(int)
        )

        dow_weights = self._day_of_week_analysis(df_est)
        self._hourly_analysis(df_est)
        self._inactivity_analysis(df)
        self._weekly_summary(df_est)
        return dow_weights

    def _day_of_week_analysis(self, df: pd.DataFrame) -> dict:
        dow_totals  = df.groupby("DayOfWeek")["DailyCount"].sum()
        grand_total = dow_totals.sum()
        logger.info("=" * 60)
        logger.info("TEMPORAL PATTERN — Day-of-Week Distribution (EST)")
        logger.info("=" * 60)
        weights = {}
        for d in range(7):
            count  = int(dow_totals.get(d, 0))
            weight = count / grand_total if grand_total > 0 else 1 / 7
            weights[d] = weight
            bar = "█" * int(weight * 40)
            logger.info(
                "  %-9s │ %s %.1f%%  (%d tweets)",
                self.DAY_NAMES[d], bar.ljust(40), weight * 100, count,
            )
        logger.info("=" * 60)
        return weights

    def _hourly_analysis(self, df: pd.DataFrame) -> None:
        hourly    = df.groupby("Hour")["DailyCount"].sum()
        peak_hour = int(hourly.idxmax()) if not hourly.empty else -1
        logger.info("TEMPORAL PATTERN — Hourly Distribution (EST)")
        logger.info("-" * 60)
        for h in range(24):
            count = int(hourly.get(h, 0))
            bar   = "▪" * min(count, 50)
            logger.info("  %02d:00 │ %s %d", h, bar.ljust(50), count)
        logger.info("  ► Peak hour (EST): %02d:00", peak_hour)
        logger.info("-" * 60)

    def _inactivity_analysis(self, df: pd.DataFrame) -> None:
        if len(df) < 2:
            return
        gaps = df["DateTime_UTC"].diff().dropna().dt.total_seconds() / 3600
        logger.info("TEMPORAL PATTERN — Inactivity Gap Statistics")
        logger.info("-" * 60)
        logger.info("  Mean gap              : %.2f hours", gaps.mean())
        logger.info("  Median gap            : %.2f hours", gaps.median())
        logger.info("  Max gap (longest)     : %.2f hours", gaps.max())
        logger.info("  Std dev of gaps       : %.2f hours", gaps.std())
        threshold = gaps.mean() + 2 * gaps.std()
        long_gaps = gaps[gaps > threshold]
        logger.info(
            "  Unusually long (>%.1f hrs): %d occurrences",
            threshold, len(long_gaps),
        )
        logger.info("-" * 60)

    def _weekly_summary(self, df: pd.DataFrame) -> None:
        df = df.copy()
        df["Week"] = df["DateTime_EST"].dt.isocalendar().week.astype(int)
        df["Year"] = df["DateTime_EST"].dt.year
        weekly     = df.groupby(["Year", "Week"])["DailyCount"].sum()
        if weekly.empty:
            return
        logger.info("TEMPORAL PATTERN — Weekly Tweet Summary")
        logger.info("-" * 60)
        for (yr, wk), total in weekly.items():
            logger.info("    %d-W%02d │ %d tweets", yr, wk, int(total))
        logger.info(
            "  Mean: %.1f  |  Std: %.1f  |  Min: %d  |  Max: %d",
            weekly.mean(), weekly.std(),
            int(weekly.min()), int(weekly.max()),
        )
        logger.info("-" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# DEVIATION DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

class DeviationDetector:
    """
    Compares actual daily tweet count against the expected daily count
    derived from posterior_mean x day-of-week weight and returns a Z-score.
    Triggers a news scan when |Z| >= DEVIATION_Z_THRESHOLD.
    """

    def __init__(self, csv_manager: CSVManager):
        self.csv = csv_manager
        self._historical_daily: list[float] = []
        self._refresh_history()

    def _refresh_history(self) -> None:
        df = self.csv.load_dataframe()
        if df.empty or len(df) < 2:
            return
        df["DateTime_EST"] = df["DateTime_UTC"].dt.tz_convert(EST_TZ)
        df["Date_EST"]     = df["DateTime_EST"].dt.date
        df["DailyCount"]   = (
            df["Cumulative_Tweet_Count"].diff()
            .fillna(df["Cumulative_Tweet_Count"].iloc[0])
            .clip(lower=0)
        )
        self._historical_daily = (
            df.groupby("Date_EST")["DailyCount"].sum().tolist()
        )

    def compute_z_score(self, actual: int, expected: float) -> float:
        self._refresh_history()
        if len(self._historical_daily) >= 5:
            std = max(float(np.std(self._historical_daily)), 1.0)
            return (actual - expected) / std
        if expected > 0:
            return (actual - expected) / max(expected * 0.3, 1.0)
        return 0.0

    def evaluate(
        self,
        day_of_week:    int,
        actual_count:   int,
        posterior_mean: float,
        day_weights:    dict,
    ) -> dict:
        weight         = day_weights.get(day_of_week, 1 / 7)
        expected_daily = posterior_mean * weight
        z_score        = self.compute_z_score(actual_count, expected_daily)
        abs_z          = abs(z_score)
        is_deviation   = abs_z >= DEVIATION_Z_THRESHOLD
        day_name       = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_of_week]

        result = {
            "day":                 day_name,
            "actual_count":        actual_count,
            "expected_count":      round(expected_daily, 1),
            "z_score":             round(z_score, 3),
            "abs_z_score":         round(abs_z, 3),
            "is_high_deviation":   is_deviation,
            "deviation_direction": "ABOVE" if z_score > 0 else "BELOW",
        }

        if is_deviation:
            logger.warning(
                "⚠️  DEVIATION on %s — actual: %d  expected: %.1f  "
                "Z=%.2f (%s) → news scan triggered",
                day_name, actual_count, expected_daily,
                z_score, result["deviation_direction"],
            )
        else:
            logger.info(
                "✅ Normal activity on %s — actual: %d  expected: %.1f  Z=%.2f",
                day_name, actual_count, expected_daily, z_score,
            )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# NEWS SCANNER
# ──────────────────────────────────────────────────────────────────────────────

class NewsScanner:
    def __init__(self, session: aiohttp.ClientSession):
        self._session      = session
        self._api_key      = NEWS_API_KEY
        self._last_scan_at: Optional[datetime] = None

        if not self._api_key or self._api_key == "your_newsapi_key_here":
            logger.warning(
                "NEWS_API_KEY is a placeholder — replace it in CONFIGURATION."
            )

    async def fetch(
        self,
        lookback_hours: int = NEWS_LOOKBACK_HOURS,
        categories:     Optional[list[str]] = None,
    ) -> list[dict]:
        if not self._api_key or self._api_key == "your_newsapi_key_here":
            logger.warning("Skipping news scan: NEWS_API_KEY not set.")
            return []

        categories = categories or list(EVENT_TAXONOMY.keys())
        from_str   = (
            datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        all_articles: list[dict] = []
        seen_urls:    set[str]   = set()

        for category in categories:
            taxonomy = EVENT_TAXONOMY[category]
            query    = " OR ".join(f'"{q}"' for q in taxonomy["queries"])
            params   = {
                "q":        query,
                "from":     from_str,
                "sortBy":   "publishedAt",
                "language": "en",
                "pageSize": MAX_ARTICLES_PER_SCAN,
                "apiKey":   self._api_key,
            }

            try:
                async with self._session.get(
                    NEWS_API_BASE_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 429:
                        logger.warning(
                            "NewsAPI rate limit — skipping: %s", category
                        )
                        continue
                    resp.raise_for_status()
                    data = await resp.json()

                for art in data.get("articles", []):
                    url = art.get("url", "")
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    all_articles.append({
                        "title":       art.get("title", ""),
                        "description": art.get("description", "") or "",
                        "publishedAt": art.get("publishedAt", ""),
                        "source":      art.get("source", {}).get("name", ""),
                        "url":         url,
                        "category":    category,
                    })

                logger.info(
                    "NewsAPI %-16s : %d articles",
                    category, len(data.get("articles", [])),
                )
                await asyncio.sleep(0.3)

            except asyncio.TimeoutError:
                logger.warning("NewsAPI timeout: %s", category)
            except Exception as exc:
                logger.error("NewsAPI error (%s): %s", category, exc)

        self._last_scan_at = datetime.now(timezone.utc)
        logger.info(
            "News scan complete — %d unique articles.", len(all_articles)
        )
        return all_articles


# ──────────────────────────────────────────────────────────────────────────────
# EVENT FACTOR ANALYZER
# ──────────────────────────────────────────────────────────────────────────────

class EventFactorAnalyzer:
    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    def analyze(self, articles: list[dict]) -> dict:
        if not articles:
            return self._empty_result()

        category_accumulator: dict[str, list[float]] = defaultdict(list)
        article_scores:       list[dict]             = []

        for article in articles:
            category  = article["category"]
            taxonomy  = EVENT_TAXONOMY[category]
            direction = taxonomy["direction"]
            weight    = taxonomy["weight"]

            text      = f"{article['title']} {article['description']}".lower()
            kw_hits   = sum(1 for kw in taxonomy["keywords"] if kw in text)
            relevance = min(
                kw_hits / max(len(taxonomy["keywords"]) * 0.25, 1), 1.0
            )
            sentiment = self._vader.polarity_scores(
                f"{article['title']} {article['description']}"
            )["compound"]

            signed_score = (
                relevance * direction * (0.5 + 0.5 * abs(sentiment)) * weight
            )
            category_accumulator[category].append(signed_score)
            article_scores.append({
                "title":        article["title"],
                "source":       article["source"],
                "category":     category,
                "relevance":    round(relevance, 3),
                "sentiment":    round(sentiment, 3),
                "signed_score": round(signed_score, 4),
                "url":          article["url"],
            })

        category_scores: dict[str, float] = {}
        total_score = 0.0
        for cat, scores in category_accumulator.items():
            cat_score            = float(np.sum(scores))
            category_scores[cat] = round(cat_score, 4)
            total_score         += cat_score

        n_active     = max(len(category_accumulator), 1)
        raw_factor   = total_score / n_active
        event_factor = float(
            np.clip(raw_factor, -EVENT_FACTOR_MAX, EVENT_FACTOR_MAX)
        )

        article_scores.sort(key=lambda x: abs(x["signed_score"]), reverse=True)

        result = {
            "event_factor":    round(event_factor, 4),
            "raw_factor":      round(raw_factor, 4),
            "category_scores": category_scores,
            "top_articles":    article_scores[:5],
            "total_articles":  len(articles),
            "scan_time_utc":   datetime.now(timezone.utc).isoformat(),
        }
        self._log_result(result)
        return result

    def _log_result(self, result: dict) -> None:
        ef   = result["event_factor"]
        sign = "+" if ef >= 0 else ""
        event_logger.info("=" * 70)
        event_logger.info(
            "EVENT FACTOR UPDATE → %s%.4f  (%s%.1f%% weekly adjustment)",
            sign, ef, sign, ef * 100,
        )
        event_logger.info("  Articles analyzed: %d", result["total_articles"])
        for cat, score in result["category_scores"].items():
            bar = ("▲" if score >= 0 else "▼") * min(int(abs(score) * 20), 20)
            event_logger.info(
                "    %-18s │ %s%.4f  %s",
                cat, "+" if score >= 0 else "", score, bar,
            )
        for i, art in enumerate(result["top_articles"], 1):
            event_logger.info(
                "    %d. [%s] %-50s  score=%+.4f  sentiment=%+.3f",
                i, art["category"].upper()[:12], art["title"][:50],
                art["signed_score"], art["sentiment"],
            )
        event_logger.info("=" * 70)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "event_factor":    0.0,
            "raw_factor":      0.0,
            "category_scores": {},
            "top_articles":    [],
            "total_articles":  0,
            "scan_time_utc":   datetime.now(timezone.utc).isoformat(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# EVENT FACTOR TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class EventFactorTracker:
    def __init__(self):
        self.current_factor: float         = 0.0
        self.factor_history: list[dict]    = []
        self._last_updated:  Optional[str] = None

    def update(self, new_factor: float, source: str = "scan") -> float:
        alpha               = 0.6
        self.current_factor = (
            alpha * new_factor + (1 - alpha) * self.current_factor
        )
        self.current_factor = float(
            np.clip(self.current_factor, -EVENT_FACTOR_MAX, EVENT_FACTOR_MAX)
        )
        self._last_updated = datetime.now(timezone.utc).isoformat()
        self.factor_history.append({
            "timestamp":  self._last_updated,
            "new_factor": round(new_factor, 4),
            "blended":    round(self.current_factor, 4),
            "source":     source,
        })
        self.factor_history = self.factor_history[-500:]
        event_logger.info(
            "EventFactorTracker — new=%.4f  blended=%.4f  source=%s",
            new_factor, self.current_factor, source,
        )
        return self.current_factor

    def decay(self) -> float:
        before              = self.current_factor
        self.current_factor *= EVENT_FACTOR_DECAY
        if abs(self.current_factor) < 0.005:
            self.current_factor = 0.0
        event_logger.info(
            "EventFactor decay: %.4f → %.4f", before, self.current_factor
        )
        return self.current_factor

    def adjusted_prediction(self, bayesian_mean: float) -> int:
        return max(0, round(bayesian_mean * (1.0 + self.current_factor)))

    def summary(self) -> str:
        sign = "+" if self.current_factor >= 0 else ""
        return (
            f"EventFactor={sign}{self.current_factor:.4f} "
            f"({sign}{self.current_factor * 100:.1f}% adjustment) "
            f"last_updated={self._last_updated}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# HAWKES PROCESS MODEL  (self-exciting point process, exponential kernel)
# ──────────────────────────────────────────────────────────────────────────────

class HawkesProcessModel:
    """
    Exponential-kernel Hawkes process fitted by maximum likelihood on the
    historical tweet timestamps.

        λ(t) = μ + α · Σ_{t_i < t} exp(−β (t − t_i))

    μ       — baseline (exogenous) intensity            [tweets / hour]
    α       — excitation jump added by each tweet       [1 / hour]
    β       — exponential decay rate of the excitation  [1 / hour]
    n = α/β — branching ratio (mean number of "child" tweets each tweet
              triggers); the process is stationary only when n < 1.

    The log-likelihood over [0, T] uses the O(N) Ozaki recursion
    A(i) = e^{−β Δt_i} (1 + A(i−1)):

        ℓ(μ,α,β) = Σ_i log(μ + α A(i)) − μT − (α/β) Σ_i (1 − e^{−β(T−t_i)})

    maximised over (log μ, log α, log β) with L-BFGS-B (multi-start on β).

    Conditional forecasting uses the exact conditional expectation of the
    exponential Hawkes process: with λ∞ = μ/(1−n) and κ = β − α,

        E[λ(t+s) | F_t]    = λ∞ + (λ(t) − λ∞) e^{−κ s}
        E[N(t, t+h] | F_t] = λ∞ h + (λ(t) − λ∞)(1 − e^{−κ h}) / κ

    This is exactly the amplify/decay behaviour required:
      • during a burst λ(t) ≫ λ∞ → the near-term forecast is amplified,
        relaxing back to the stationary rate at speed κ;
      • after a long silence λ(t) ≈ μ < λ∞ → the forecast is suppressed.
    Count overdispersion uses the stationary Fano factor 1/(1−n)².
    """

    def __init__(self):
        self.mu:              float = 0.0
        self.alpha:           float = 0.0
        self.beta:            float = 1.0
        self.branching_ratio: float = 0.0
        self.log_likelihood:  float = float("-inf")
        self.n_events_fit:    int   = 0
        self.fitted:          bool  = False

    @staticmethod
    def _neg_log_likelihood(log_params: np.ndarray, t: np.ndarray, T: float) -> float:
        mu, alpha, beta = np.exp(log_params)
        if not np.isfinite(mu + alpha + beta) or beta <= 0:
            return 1e12
        n  = len(t)
        dt = np.diff(t)
        decay = np.exp(-beta * dt)
        A = np.empty(n)
        A[0] = 0.0
        for i in range(1, n):
            A[i] = decay[i - 1] * (1.0 + A[i - 1])
        lam = mu + alpha * A
        if np.any(lam <= 1e-12):
            return 1e12
        compensator = mu * T + (alpha / beta) * float(
            np.sum(1.0 - np.exp(-beta * (T - t)))
        )
        ll = float(np.sum(np.log(lam))) - compensator
        # Soft barrier keeping the process subcritical (n = α/β < 1)
        br = alpha / beta
        penalty = 0.0 if br < 0.95 else 1e4 * (br - 0.95) ** 2 * n
        return -ll + penalty

    def fit(self, event_times_hours: np.ndarray) -> bool:
        t = np.sort(np.asarray(event_times_hours, dtype=float))
        t = t[np.isfinite(t)]
        if len(t) < HAWKES_MIN_EVENTS:
            return False
        if len(t) > HAWKES_MAX_FIT_EVENTS:
            t = t[-HAWKES_MAX_FIT_EVENTS:]
        t = t - t[0]
        # Enforce strictly increasing times (API can report ties)
        t = t + np.arange(len(t)) * 1e-6
        T = float(t[-1]) + 1e-3
        base_rate = len(t) / T

        best = None
        for beta0 in (0.5, 2.0, 8.0):           # multi-start on decay scale
            x0 = np.log([
                max(0.5 * base_rate, 1e-4),     # μ₀: half the empirical rate
                max(0.5 * beta0,     1e-4),     # α₀: branching ratio 0.5
                beta0,
            ])
            try:
                res = minimize(
                    self._neg_log_likelihood, x0, args=(t, T),
                    method="L-BFGS-B",
                    options={"maxiter": 150, "ftol": 1e-8},
                )
            except Exception:
                continue
            if not np.isfinite(res.fun):
                continue
            if best is None or res.fun < best.fun:
                best = res
        if best is None:
            return False

        mu, alpha, beta = (float(v) for v in np.exp(best.x))
        br = alpha / beta
        if br >= HAWKES_MAX_BRANCHING:
            br    = HAWKES_MAX_BRANCHING
            alpha = br * beta

        self.mu, self.alpha, self.beta = mu, alpha, beta
        self.branching_ratio = br
        self.log_likelihood  = -float(best.fun)
        self.n_events_fit    = len(t)
        self.fitted          = True
        return True

    def excitation_at(
        self, event_times_hours: Optional[np.ndarray], now_h: float
    ) -> float:
        """R(t) = α Σ exp(−β (t − t_i)) — the residual excitation now."""
        if not self.fitted or event_times_hours is None or len(event_times_hours) == 0:
            return 0.0
        dt = now_h - event_times_hours
        dt = dt[(dt >= 0.0) & (self.beta * dt < 40.0)]
        if len(dt) == 0:
            return 0.0
        return float(self.alpha * np.sum(np.exp(-self.beta * dt)))

    def stationary_rate(self) -> float:
        """λ∞ = μ / (1 − n) — long-run mean intensity (tweets/hour)."""
        if not self.fitted:
            return 0.0
        return self.mu / max(1.0 - self.branching_ratio, 1e-3)

    def fano_factor(self) -> float:
        """Stationary count overdispersion Var/Mean = 1/(1−n)², capped."""
        if not self.fitted:
            return 1.0
        return float(np.clip(
            1.0 / max(1.0 - self.branching_ratio, 1e-3) ** 2,
            1.0, HAWKES_FANO_MAX,
        ))

    def expected_count(self, horizon_h: float, current_excitation: float) -> float:
        """E[N(t, t+h] | F_t] — exact conditional expectation (see class doc)."""
        if not self.fitted or horizon_h <= 0:
            return 0.0
        lam_inf = self.stationary_rate()
        lam_now = self.mu + current_excitation
        kappa   = max(self.beta - self.alpha, 1e-4)
        e_n = lam_inf * horizon_h + (
            (lam_now - lam_inf) * (1.0 - np.exp(-kappa * horizon_h)) / kappa
        )
        return float(max(e_n, 0.0))


# ──────────────────────────────────────────────────────────────────────────────
# INTER-ARRIVAL REGIME MODEL  (latent-state silence conditioning)
# ──────────────────────────────────────────────────────────────────────────────

class InterArrivalRegimeModel:
    """
    Latent-regime model of inter-tweet gaps: a two-component exponential
    (hyperexponential) mixture fitted with EM,

        f(τ) = π₁ r₁ e^{−r₁ τ} + π₂ r₂ e^{−r₂ τ}        (r₁ > r₂)

    where component 1 is the "active" regime (short gaps) and component 2
    the "dormant" regime (long sleeps / offline periods).

    Conditioning on an elapsed silence of effective length τ uses the
    survival posterior  P(regime k | gap ≥ τ) ∝ π_k e^{−r_k τ},  giving the
    instantaneous hazard of the next tweet:

        h(τ) = (π₁ r₁ e^{−r₁τ} + π₂ r₂ e^{−r₂τ}) / (π₁ e^{−r₁τ} + π₂ e^{−r₂τ})

    h(τ) is monotonically *decreasing* in τ: the longer Elon has been
    silent, the more posterior mass shifts to the dormant regime, and the
    lower the expected near-term tweet rate. Because the caller measures τ
    in activity-weighted (diurnal-profile) hours, the same clock-time
    silence during a normally high-activity period counts as *stronger*
    evidence of dormancy than silence at 4 AM — exactly the conditional
    question "no tweets for a few hours during a usually busy period".

    silence_factor() returns  h(τ) / (1/E[τ])  — hazard now relative to the
    unconditional long-run event rate — clipped to a sane range.
    """

    def __init__(self):
        self.pi1:       float = 0.0
        self.r1:        float = 0.0
        self.r2:        float = 0.0
        self.mean_rate: float = 0.0
        self.fitted:    bool  = False

    def fit(self, gaps_hours: np.ndarray) -> bool:
        x = np.asarray(gaps_hours, dtype=float)
        x = x[np.isfinite(x) & (x > 1e-4)]
        if len(x) < HAWKES_MIN_EVENTS:
            return False

        r1  = 1.0 / max(float(np.percentile(x, 25)), 1e-3)
        r2  = 1.0 / max(float(np.percentile(x, 90)), 1e-2)
        if r1 <= r2:
            r1 = 4.0 * r2
        pi1 = 0.6

        for _ in range(300):                     # EM iterations
            d1 = pi1 * r1 * np.exp(-np.minimum(r1 * x, 700.0))
            d2 = (1.0 - pi1) * r2 * np.exp(-np.minimum(r2 * x, 700.0))
            w  = d1 / (d1 + d2 + 1e-300)
            new_pi1 = float(np.clip(w.mean(), 0.01, 0.99))
            new_r1  = float(np.sum(w) / max(np.sum(w * x), 1e-12))
            new_r2  = float(np.sum(1.0 - w) / max(np.sum((1.0 - w) * x), 1e-12))
            converged = (
                abs(new_pi1 - pi1) < 1e-8
                and abs(new_r1 - r1) < 1e-8
                and abs(new_r2 - r2) < 1e-8
            )
            pi1, r1, r2 = new_pi1, new_r1, new_r2
            if converged:
                break

        if r1 < r2:                              # component 1 = "active"
            r1, r2 = r2, r1
            pi1    = 1.0 - pi1

        self.pi1, self.r1, self.r2 = pi1, r1, r2
        self.mean_rate = 1.0 / max(pi1 / r1 + (1.0 - pi1) / r2, 1e-9)
        self.fitted    = True
        return True

    def silence_factor(self, tau_hours: float) -> float:
        if not self.fitted or tau_hours < 0:
            return 1.0
        e1   = self.pi1 * np.exp(-min(self.r1 * tau_hours, 700.0))
        e2   = (1.0 - self.pi1) * np.exp(-min(self.r2 * tau_hours, 700.0))
        surv = e1 + e2
        hazard = self.r2 if surv < 1e-300 else (e1 * self.r1 + e2 * self.r2) / surv
        return float(np.clip(
            hazard / max(self.mean_rate, 1e-9),
            SILENCE_FACTOR_MIN, SILENCE_FACTOR_MAX,
        ))


# ──────────────────────────────────────────────────────────────────────────────
# CONDITIONAL FORECAST ENGINE  (Hawkes + regime + diurnal + Gamma–Poisson)
# ──────────────────────────────────────────────────────────────────────────────

class ConditionalForecastEngine:
    """
    Combines four components into one conditional forecast of future
    tweet counts for an arbitrary horizon:

      1. HawkesProcessModel — burst amplification / post-burst decay via
         the exact conditional expectation of the self-exciting intensity;
      2. a diurnal × weekly activity profile ρ(hour, dow) (mean 1),
         estimated from history, used to convert clock time into
         "activity-weighted exposure" (effective hours). The Hawkes model
         is fitted and evaluated entirely on this effective-time axis:
         quiet hours pass "slowly" on it, so tweets posted during normally
         quiet hours sit closer together and generate *stronger* excitation
         than the same burst at peak hours, and the diurnal/weekly shape of
         the remaining horizon is handled by integrating ρ over it rather
         than by a flat suppressive multiplier;
      3. InterArrivalRegimeModel — silence conditioning through the gap
         survival posterior (evaluated at activity-weighted elapsed time);
      4. conjugate Gamma–Poisson calibration: a window-specific rate
         multiplier θ with prior Gamma(a₀, a₀) updated by the tweets
         already observed in the bet window, posterior mean
         θ̂ = (a₀ + observed) / (a₀ + expected-so-far),
         whose posterior uncertainty is propagated into the predictive
         variance together with the Hawkes Fano factor (a negative-
         binomial-style overdispersed predictive distribution).
    """

    def __init__(self, csv_manager: CSVManager):
        self.csv    = csv_manager
        self.hawkes = HawkesProcessModel()
        self.regime = InterArrivalRegimeModel()
        self._rho_hour: np.ndarray = np.ones(24)
        self._rho_dow:  np.ndarray = np.ones(7)
        self._event_epoch_s: Optional[np.ndarray] = None
        self._event_eff_h:   Optional[np.ndarray] = None
        self._epoch0_s:    float              = 0.0
        self._last_fit_at: Optional[datetime] = None
        self._events_refreshed_at: Optional[datetime] = None
        self._fit_lock = threading.Lock()

    @property
    def is_fitted(self) -> bool:
        return (
            self.hawkes.fitted
            and self._event_epoch_s is not None
            and len(self._event_epoch_s) > 0
            and self._event_eff_h is not None
        )

    # ── Fitting ───────────────────────────────────────────────────────────

    def refit(self, force: bool = False) -> bool:
        """
        Refreshes the live event arrays (cheap — every call) and refits the
        sub-models (expensive MLE — throttled to HAWKES_REFIT_INTERVAL
        unless force=True). All fitting happens on the activity-rescaled
        ("effective") time axis. Safe to call from a worker thread
        (non-blocking lock, diagnostics at DEBUG level).
        """
        now = datetime.now(timezone.utc)
        fit_due = (
            force
            or self._last_fit_at is None
            or (now - self._last_fit_at).total_seconds() >= HAWKES_REFIT_INTERVAL
        )
        if not self._fit_lock.acquire(blocking=False):
            return self.is_fitted
        try:
            df = self.csv.load_dataframe()
            if df.empty or len(df) < HAWKES_MIN_EVENTS:
                logger.debug(
                    "Hawkes refit skipped — only %d events in history.", len(df)
                )
                return False

            epoch_s = np.sort(
                df["DateTime_UTC"].astype("int64").to_numpy() / 1e9
            ).astype(float)

            # Diurnal × weekly activity profile (Laplace-smoothed, mean ≈ 1)
            est = df["DateTime_UTC"].dt.tz_convert(EST_TZ)
            hour_counts = (
                np.bincount(est.dt.hour.to_numpy(), minlength=24)
                .astype(float) + 1.0
            )
            dow_counts = (
                np.bincount(est.dt.dayofweek.to_numpy(), minlength=7)
                .astype(float) + 1.0
            )
            self._rho_hour = np.clip(hour_counts / hour_counts.mean(), 0.05, None)
            self._rho_dow  = np.clip(dow_counts / dow_counts.mean(),  0.05, None)

            # Effective (activity-rescaled) event coordinates: ∫ρ ds up to
            # each tweet. Tweets in normally quiet hours sit closer together
            # on this axis, so the excitation they generate is expressed
            # MORE strongly instead of being diluted by the low diurnal
            # weight of the surrounding hours.
            eff_h = self._effective_event_hours(epoch_s)

            self._event_epoch_s       = epoch_s
            self._event_eff_h         = eff_h
            self._epoch0_s            = float(epoch_s[0])
            self._events_refreshed_at = now

            if not fit_due:
                return self.is_fitted

            hawkes_ok = self.hawkes.fit(eff_h)

            gaps = np.diff(eff_h)
            self.regime.fit(gaps[gaps > 0])

            self._last_fit_at = now

            if hawkes_ok:
                logger.debug(
                    "Hawkes refit ok (effective-time axis) — mu=%.4f/h "
                    "alpha=%.4f beta=%.4f n=%.3f lam_inf=%.3f/h Fano=%.2f "
                    "events=%d regime=(pi1=%.2f r1=%.3f r2=%.3f)",
                    self.hawkes.mu, self.hawkes.alpha, self.hawkes.beta,
                    self.hawkes.branching_ratio, self.hawkes.stationary_rate(),
                    self.hawkes.fano_factor(), self.hawkes.n_events_fit,
                    self.regime.pi1, self.regime.r1, self.regime.r2,
                )
            else:
                logger.debug(
                    "Hawkes MLE did not converge — Bayesian blend fallback active."
                )
            return self.is_fitted
        except Exception as exc:
            logger.debug("Hawkes refit error: %s", exc)
            return self.is_fitted
        finally:
            self._fit_lock.release()

    def _effective_event_hours(self, epoch_s: np.ndarray) -> np.ndarray:
        """∫ρ(s) ds from the first event to each event — event times on the
        activity-weighted clock (hours, mean weight ≈ 1)."""
        t0      = float(epoch_s[0])
        total_h = max((float(epoch_s[-1]) - t0) / 3600.0, 0.0)
        step    = 0.5
        grid_s  = t0 + np.arange(int(total_h / step) + 2) * (step * 3600.0)
        grid_est = pd.to_datetime(grid_s, unit="s", utc=True).tz_convert(EST_TZ)
        rho = np.clip(
            self._rho_hour[grid_est.hour.to_numpy()]
            * self._rho_dow[grid_est.dayofweek.to_numpy()],
            0.05, 4.0,
        )
        cum = np.concatenate(([0.0], np.cumsum(rho[:-1]) * step))
        return np.interp(epoch_s, grid_s, cum)

    def _refresh_events(self) -> None:
        """
        Cheap staleness guard (no MLE): reloads the event arrays from the
        CSV when they are older than EVENT_REFRESH_MAX_AGE_S, so the
        excitation seen by the forecast includes tweets ingested since the
        last throttled refit. Previously the excitation was computed from
        arrays up to 30 minutes stale, so with a fast-decaying fitted
        kernel a live burst could decay away before it was ever expressed
        in a prediction.
        """
        if self._events_refreshed_at is not None and (
            datetime.now(timezone.utc) - self._events_refreshed_at
        ).total_seconds() < EVENT_REFRESH_MAX_AGE_S:
            return
        if not self._fit_lock.acquire(blocking=False):
            return
        try:
            df = self.csv.load_dataframe()
            if df.empty:
                return
            epoch_s = np.sort(
                df["DateTime_UTC"].astype("int64").to_numpy() / 1e9
            ).astype(float)
            self._event_epoch_s       = epoch_s
            self._event_eff_h         = self._effective_event_hours(epoch_s)
            self._epoch0_s            = float(epoch_s[0])
            self._events_refreshed_at = datetime.now(timezone.utc)
        except Exception as exc:
            logger.debug("Event refresh error: %s", exc)
        finally:
            self._fit_lock.release()

    # ── Activity profile helpers ──────────────────────────────────────────

    def _rho(self, hour: int, dow: int) -> float:
        return float(np.clip(
            self._rho_hour[hour] * self._rho_dow[dow], 0.05, 4.0
        ))

    def _weighted_hours(self, t0: datetime, t1: datetime) -> float:
        """∫ρ(s) ds between t0 and t1 — clock time converted to
        activity-weighted exposure ('effective hours', mean weight 1)."""
        total_clock = (t1 - t0).total_seconds() / 3600.0
        if total_clock <= 0:
            return 0.0
        total_clock = min(total_clock, FORECAST_HORIZON_CAP_H)
        step = 0.5
        acc  = 0.0
        x    = 0.0
        while x < total_clock:
            frac = min(step, total_clock - x)
            mid  = (t0 + timedelta(hours=x + frac / 2.0)).astimezone(EST_TZ)
            acc += frac * self._rho(mid.hour, mid.weekday())
            x   += step
        return acc

    # ── Conditional forecasts ─────────────────────────────────────────────

    def forecast_window(
        self,
        count_in_window: int,
        start_dt:        datetime,
        end_dt:          datetime,
        now:             Optional[datetime] = None,
    ) -> Optional[tuple[float, float, float]]:
        """
        Conditional forecast of the total tweet count for a bet window:
        returns (point, ci95_lower, ci95_upper) or None when unavailable.
        """
        if not self.is_fitted:
            return None
        try:
            now = now or datetime.now(timezone.utc)
            remaining_h = (end_dt - now).total_seconds() / 3600.0
            if remaining_h <= 0:
                c = float(count_in_window)
                return c, c, c

            self._refresh_events()
            hk = self.hawkes

            # Everything below lives on the effective (activity-weighted)
            # time axis the Hawkes model was fitted on, so excitation,
            # horizon and silence conditioning are mutually consistent.
            last_dt = datetime.fromtimestamp(
                float(self._event_epoch_s[-1]), tz=timezone.utc
            )
            gap_eff = max(self._weighted_hours(last_dt, now), 0.0)
            tau_now = float(self._event_eff_h[-1]) + gap_eff
            excitation = hk.excitation_at(self._event_eff_h, tau_now)

            # 1. Hawkes conditional expectation over the remaining horizon,
            #    measured in effective hours — the diurnal/weekly shape of
            #    the horizon is inherent in the axis (no flat ρ̄ multiplier,
            #    which used to dilute quiet-hour bursts); beyond the
            #    numerical cap the excitation has fully decayed, so the
            #    residual accrues at the stationary rate.
            h_eff  = self._weighted_hours(now, end_dt)
            e_base = hk.expected_count(h_eff, excitation)
            if remaining_h > FORECAST_HORIZON_CAP_H:
                e_base += hk.stationary_rate() * (
                    remaining_h - FORECAST_HORIZON_CAP_H
                )

            # 2. Silence conditioning: hazard given the activity-weighted
            #    gap since the last tweet (regime survival posterior)
            silence = self.regime.silence_factor(gap_eff)

            e_adj = max(e_base * silence, 0.0)

            # 3. Gamma–Poisson window calibration:
            #    θ | window ~ Gamma(a₀ + observed, a₀ + expected-so-far)
            elapsed_eff     = self._weighted_hours(start_dt, now)
            expected_so_far = hk.stationary_rate() * elapsed_eff
            theta = (WINDOW_CALIB_PRIOR + count_in_window) / (
                WINDOW_CALIB_PRIOR + max(expected_so_far, 1e-6)
            )
            theta   = float(np.clip(theta, THETA_MIN, THETA_MAX))
            e_final = e_adj * theta

            # Predictive variance: Hawkes overdispersion (Fano factor)
            # plus posterior uncertainty of the calibration multiplier θ
            fano = hk.fano_factor()
            var  = e_final * fano + (e_final ** 2) / (
                WINDOW_CALIB_PRIOR + count_in_window
            )
            std  = float(np.sqrt(max(var, 1.0)))

            wp  = float(max(count_in_window,
                            round(count_in_window + e_final)))
            wlo = float(max(count_in_window, round(wp - 1.96 * std)))
            whi = float(round(wp + 1.96 * std))
            return wp, wlo, whi

        except Exception as exc:
            logger.debug("forecast_window error: %s", exc)
            return None

    def weekly_total_estimate(
        self, cumulative_so_far: int
    ) -> Optional[tuple[float, float]]:
        """
        Hawkes-conditional estimate of the current Mon–Sun (EST) weekly
        total: observed-so-far plus the conditional expectation of the
        remainder. Returns (mean, variance) or None when unavailable.
        """
        if not self.is_fitted:
            return None
        try:
            now     = datetime.now(timezone.utc)
            now_est = now.astimezone(EST_TZ)
            days_ahead   = 7 - now_est.weekday()
            week_end_est = EST_TZ.localize(datetime.combine(
                (now_est + timedelta(days=days_ahead)).date(),
                datetime.min.time(),
            ))
            week_end    = week_end_est.astimezone(timezone.utc)
            remaining_h = max((week_end - now).total_seconds() / 3600.0, 0.0)
            if remaining_h <= 0:
                return float(cumulative_so_far), 25.0

            self._refresh_events()
            hk = self.hawkes
            last_dt = datetime.fromtimestamp(
                float(self._event_epoch_s[-1]), tz=timezone.utc
            )
            gap_eff = max(self._weighted_hours(last_dt, now), 0.0)
            tau_now = float(self._event_eff_h[-1]) + gap_eff
            excitation = hk.excitation_at(self._event_eff_h, tau_now)
            h_eff   = self._weighted_hours(now, week_end)
            e_base  = hk.expected_count(h_eff, excitation)
            silence = self.regime.silence_factor(gap_eff)
            e_rem = max(e_base * silence, 0.0)
            var   = max(e_rem * hk.fano_factor(), 25.0)
            return float(cumulative_so_far + e_rem), float(var)

        except Exception as exc:
            logger.debug("weekly_total_estimate error: %s", exc)
            return None


# Module-level engine handle — wired up in main(); every consumer falls
# back to the original Bayesian blend whenever this is None / unfitted.
FORECAST_ENGINE: Optional[ConditionalForecastEngine] = None


# ──────────────────────────────────────────────────────────────────────────────
# BAYESIAN TWEET FORECASTER
# ──────────────────────────────────────────────────────────────────────────────

class BayesianTweetForecaster:
    def __init__(
        self,
        prior_mean:           float = 100.0,
        prior_std:            float = 40.0,
        day_weights:          Optional[dict] = None,
        training_weeks:       Optional[list] = None,
        event_factor_tracker: Optional[EventFactorTracker] = None,
    ):
        self.prior_mean            = prior_mean
        self.prior_variance        = prior_std ** 2
        self.day_weights           = day_weights or {i: 1 / 7 for i in range(7)}
        self.training_weeks        = training_weeks or []
        self.event_tracker         = event_factor_tracker or EventFactorTracker()
        self.forecast_engine: Optional[ConditionalForecastEngine] = None
        self._reset_week_state()

    def _reset_week_state(self) -> None:
        self._observed_days:     dict[int, int] = {}
        self._posterior_mean     = self.prior_mean
        self._posterior_variance = self.prior_variance

    def update(self, day_of_week: int, tweet_count: int) -> dict:
        self._observed_days[day_of_week] = tweet_count

        implied_totals = [
            count / self.day_weights.get(d, 1 / 7)
            for d, count in self._observed_days.items()
            if self.day_weights.get(d, 1 / 7) > 0
        ]

        likelihood_mean     = float(np.mean(implied_totals))
        likelihood_variance = float(np.var(implied_totals)) + max(
            self.prior_variance * 0.05, 25.0
        )

        prec_prior      = 1.0 / self._posterior_variance
        prec_likelihood = 1.0 / likelihood_variance
        prec_posterior  = prec_prior + prec_likelihood

        self._posterior_variance = 1.0 / prec_posterior
        self._posterior_mean     = self._posterior_variance * (
            prec_prior      * self._posterior_mean +
            prec_likelihood * likelihood_mean
        )

        # ── Hawkes-conditional fusion (report-level, non-compounding) ─────
        # The Hawkes engine supplies an independent Gaussian "observation"
        # of the weekly total: tweets observed so far plus the conditional
        # expectation of the remainder (burst-amplified or silence-
        # suppressed, diurnally weighted). It is fused with the conjugate
        # posterior by precision weighting for *this report only* — the
        # stored posterior keeps evolving from daily observations alone,
        # so the Hawkes evidence is never compounded across repeated
        # update() calls within the same week.
        report_mean = self._posterior_mean
        report_var  = self._posterior_variance
        engine = self.forecast_engine
        if engine is not None and engine.is_fitted:
            try:
                hawkes_est = engine.weekly_total_estimate(
                    sum(self._observed_days.values())
                )
            except Exception:
                hawkes_est = None
            if hawkes_est is not None:
                h_mean, h_var = hawkes_est
                h_var         = max(h_var, 25.0)
                prec_post     = 1.0 / report_var
                prec_hawkes   = 1.0 / h_var
                report_var    = 1.0 / (prec_post + prec_hawkes)
                report_mean   = report_var * (
                    prec_post * report_mean + prec_hawkes * h_mean
                )

        std = float(np.sqrt(report_var))
        ef  = self.event_tracker.current_factor

        adjusted_mean  = self.event_tracker.adjusted_prediction(report_mean)
        adjusted_lower = max(
            0, round((report_mean - 1.96 * std) * (1 + ef))
        )
        adjusted_upper = round(
            (report_mean + 1.96 * std) * (1 + ef)
        )

        result = {
            "bayesian_weekly_total":  round(report_mean),
            "posterior_mean":         report_mean,
            "posterior_std":          std,
            "bayesian_ci_95_lower":   max(
                0, round(report_mean - 1.96 * std)
            ),
            "bayesian_ci_95_upper":   round(
                report_mean + 1.96 * std
            ),
            "predicted_weekly_total": adjusted_mean,
            "event_factor":           round(ef, 4),
            "event_adjustment_pct":   round(ef * 100, 1),
            "adjusted_ci_95_lower":   adjusted_lower,
            "adjusted_ci_95_upper":   adjusted_upper,
            "days_observed":          len(self._observed_days),
            "days_remaining":         7 - len(self._observed_days),
            "cumulative_so_far":      sum(self._observed_days.values()),
        }

        self._log_prediction(day_of_week, tweet_count, result)
        return result

    def retrain(
        self,
        actual_weekly_total: int,
        new_day_weights:     Optional[dict] = None,
    ) -> None:
        self.training_weeks.append(actual_weekly_total)
        if len(self.training_weeks) >= 2:
            self.prior_mean     = float(np.mean(self.training_weeks))
            self.prior_variance = float(np.var(self.training_weeks)) + 1.0
        else:
            self.prior_mean     = (self.prior_mean + actual_weekly_total) / 2
            self.prior_variance = max(self.prior_variance, 400.0)
        if new_day_weights:
            self.day_weights = new_day_weights
        logger.info(
            "MODEL RETRAINED — actual=%d  new prior: mean=%.1f  std=%.1f  "
            "training_weeks=%d",
            actual_weekly_total, self.prior_mean,
            np.sqrt(self.prior_variance), len(self.training_weeks),
        )
        self._reset_week_state()

    def save(self, filepath: str = MODEL_FILE) -> None:
        state = {
            "prior_mean":         self.prior_mean,
            "prior_variance":     self.prior_variance,
            "day_weights":        self.day_weights,
            "training_weeks":     self.training_weeks,
            "observed_days":      self._observed_days,
            "posterior_mean":     self._posterior_mean,
            "posterior_variance": self._posterior_variance,
            "event_factor":       self.event_tracker.current_factor,
            "event_history":      self.event_tracker.factor_history,
            "event_last_updated": self.event_tracker._last_updated,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info("Model saved → %s", filepath)

    @classmethod
    def load(cls, filepath: str = MODEL_FILE) -> "BayesianTweetForecaster":
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        eft                = EventFactorTracker()
        eft.current_factor = state.get("event_factor", 0.0)
        eft.factor_history = state.get("event_history", [])
        eft._last_updated  = state.get("event_last_updated")
        model = cls(
            prior_mean           = state["prior_mean"],
            prior_std            = float(np.sqrt(state["prior_variance"])),
            day_weights          = state["day_weights"],
            training_weeks       = state["training_weeks"],
            event_factor_tracker = eft,
        )
        model._observed_days      = state["observed_days"]
        model._posterior_mean     = state["posterior_mean"]
        model._posterior_variance = state["posterior_variance"]
        logger.info(
            "Model loaded ← %s  (EventFactor=%.4f)",
            filepath, eft.current_factor,
        )
        return model

    def _log_prediction(self, dow: int, count: int, result: dict) -> None:
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow]
        logger.info(
            "🔮 PREDICTION  |  %s: %d tweets  "
            "→  Bayesian: %d  |  EF=%+.4f (%+.1f%%)  "
            "→  Adjusted: %d  [95%% CI: %d–%d]  "
            "(obs: %d/7, so far: %d)",
            day_name, count,
            result["bayesian_weekly_total"],
            result["event_factor"], result["event_adjustment_pct"],
            result["predicted_weekly_total"],
            result["adjusted_ci_95_lower"], result["adjusted_ci_95_upper"],
            result["days_observed"], result["cumulative_so_far"],
        )

# ──────────────────────────────────────────────────────────────────────────────
# OPTIMAL BET CALCULATOR  (Fractional Kelly Criterion)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_optimal_bet(
    predicted_total:  int,
    ci_lower:         int,
    ci_upper:         int,
    target:           int,
    market_yes_price: float = 0.5,
    kelly_fraction:   float = 0.25,
    bankroll:         float = 100.0,
) -> dict:
    """
    Calculates an optimal bet size using Fractional Kelly Criterion.

    Estimates our model's implied probability that the tweet count will
    EXCEED the target, then compares it to the market's implied probability.

    Args:
        predicted_total:  Bayesian point estimate for weekly/window tweets
        ci_lower:         Lower bound of 95% CI
        ci_upper:         Upper bound of 95% CI
        target:           Polymarket threshold (e.g., "more than 400 tweets")
        market_yes_price: Current Polymarket YES price (e.g., 0.62 means 62 cents)
        kelly_fraction:   Fraction of full Kelly to use (0.25 = quarter-Kelly)
        bankroll:         Dollar amount to base sizing on

    Returns:
        dict with recommendation, edge, kelly %, and risk tier
    """
    if ci_upper <= ci_lower:
        return {"recommendation": "⚠️ Insufficient CI data", "bet_pct": 0.0}

    # ── Step 1: Estimate our probability using a normal approximation ──
    mean_pred = (ci_lower + ci_upper) / 2.0
    # 95% CI → sigma ≈ (upper - lower) / (2 * 1.96)
    sigma     = max((ci_upper - ci_lower) / 3.92, 1.0)

    # P(total > target) using CDF of normal distribution
    our_prob_yes = float(1.0 - norm.cdf(target, loc=mean_pred, scale=sigma))
    our_prob_yes = float(np.clip(our_prob_yes, 0.02, 0.98))  # avoid extremes

    # ── Step 2: Derive edge vs. market price ──
    market_prob = float(np.clip(market_yes_price, 0.01, 0.99))
    edge        = our_prob_yes - market_prob

    # ── Step 3: Kelly Criterion ──
    # For a binary bet at price p_market (pays 1/p_market - 1 if correct):
    #   b  = net odds (profit per \$1 wagered) = (1 - market_price) / market_price
    #   f* = (b * p_our - (1 - p_our)) / b   [full Kelly]
    b          = (1.0 - market_prob) / market_prob
    full_kelly = (b * our_prob_yes - (1.0 - our_prob_yes)) / b
    frac_kelly = kelly_fraction * full_kelly
    frac_kelly = float(np.clip(frac_kelly, 0.0, 0.30))  # hard cap at 30% bankroll

    bet_dollars = round(frac_kelly * bankroll, 2)

    # ── Step 4: Confidence tier ──
    abs_edge = abs(edge)
    if abs_edge >= 0.20:
        tier  = "🟢 STRONG edge"
        emoji = "✅"
    elif abs_edge >= 0.10:
        tier  = "🟡 MODERATE edge"
        emoji = "⚠️"
    elif abs_edge >= 0.04:
        tier  = "🟠 WEAK edge"
        emoji = "🔸"
    else:
        tier  = "🔴 NO edge / skip"
        emoji = "❌"

    # ── Step 5: Side recommendation (YES vs NO) ──
    if edge >= 0.04:
        side = "BUY YES"
    elif edge <= -0.04:
        side       = "BUY NO"
        frac_kelly = kelly_fraction * abs(full_kelly)
        frac_kelly = float(np.clip(frac_kelly, 0.0, 0.30))
        bet_dollars = round(frac_kelly * bankroll, 2)
    else:
        side = "SKIP"

    return {
        "our_prob_yes":    round(our_prob_yes * 100, 1),
        "market_prob_yes": round(market_prob  * 100, 1),
        "edge_pct":        round(edge         * 100, 1),
        "full_kelly_pct":  round(full_kelly   * 100, 1),
        "frac_kelly_pct":  round(frac_kelly   * 100, 1),
        "bet_dollars":     bet_dollars,
        "side":            side,
        "tier":            tier,
        "emoji":           emoji,
        "recommendation":  (
            f"{emoji} **{side}**  —  {tier}\n"
            f"Our P(YES): **{our_prob_yes*100:.1f}%**  vs  "
            f"Market: **{market_prob*100:.1f}%**  "
            f"(Edge: {'+' if edge>=0 else ''}{edge*100:.1f}%)\n"
            f"¼-Kelly size: **{frac_kelly*100:.1f}%** of bankroll  "
            f"(≈ ${bet_dollars:.2f} per \$100)"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# BET INTERVAL PROBABILITY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def compute_interval_probabilities(
    mean:     float,
    std:      float,
    ci_lower: float,
    ci_upper: float,
) -> list[dict]:
    """
    For every interval in ALL_INTERVALS that overlaps [ci_lower, ci_upper],
    compute P(interval) under N(mean, std²) and return a list of dicts.

    Probabilities are then re-normalised so the displayed set sums to 100 %,
    making the output self-consistent even when the tails are clipped.

    Filtering rule:
        An interval [lo, hi] is shown when  lo <= ci_upper  AND  hi >= ci_lower
        (i.e. any overlap at all with the CI range).
    """
    if std <= 0:
        std = 1.0

    relevant: list[dict] = []

    for label, lo, hi in ALL_INTERVALS:
        # Skip intervals that have NO overlap with the confidence interval
        if hi < ci_lower or lo > ci_upper:
            continue

        # Use ±0.5 continuity correction for integer-valued counts
        if label == "<20":
            # Lower tail: X < 20  →  CDF at 19.5
            prob = float(norm.cdf(hi + 0.5, loc=mean, scale=std))
        else:
            prob = float(
                norm.cdf(hi + 0.5, loc=mean, scale=std)
                - norm.cdf(lo - 0.5, loc=mean, scale=std)
            )

        relevant.append({
            "label": label,
            "low":   lo,
            "high":  hi,
            "prob":  max(prob, 0.0),
        })

    # Re-normalise over displayed intervals only
    total = sum(item["prob"] for item in relevant)
    for item in relevant:
        item["prob_pct"] = (item["prob"] / total * 100.0) if total > 0 else 0.0

    return relevant


def _compute_window_pred(
    tracking:        dict,
    prediction:      dict,
    count_in_window: int,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Replicates the blended window-prediction logic that lives inside
    DiscordNotifier so we can call it independently for console logging.

    Returns
    -------
    (wp, wlo, whi)  — point estimate and 95 % CI bounds, or
    (None, None, None) if the calculation cannot be completed.
    """
    try:
        start_r = tracking.get("startDate", "")
        end_r   = tracking.get("endDate", "")
        if not start_r or not end_r:
            return None, None, None

        pm  = prediction["posterior_mean"]
        ps  = prediction["posterior_std"]
        ef  = prediction["event_factor"]
        efm = 1.0 + ef

        now_utc        = datetime.now(timezone.utc)
        start_dt_utc   = pd.to_datetime(start_r, utc=True).to_pydatetime()
        end_dt_utc     = pd.to_datetime(end_r,   utc=True).to_pydatetime()

        # ── Hawkes-conditional forecast (preferred when engine is fitted) ──
        if FORECAST_ENGINE is not None and FORECAST_ENGINE.is_fitted:
            hawkes_fc = FORECAST_ENGINE.forecast_window(
                count_in_window = count_in_window,
                start_dt        = start_dt_utc,
                end_dt          = end_dt_utc,
                now             = now_utc,
            )
            if hawkes_fc is not None:
                return hawkes_fc

        # ── Fallback: Bayesian rate blend (engine not yet fitted) ─────────
        window_days    = max(
            (end_dt_utc - start_dt_utc).total_seconds() / 86_400, 1.0
        )
        elapsed_days   = min(
            max((now_utc - start_dt_utc).total_seconds() / 86_400, 0.0),
            window_days,
        )
        remaining_days = max(
            (end_dt_utc - now_utc).total_seconds() / 86_400, 0.0
        )

        observed_daily_rate = count_in_window / max(elapsed_days, 0.5)
        bayesian_daily_rate = (pm / 7.0) * efm
        elapsed_fraction    = min(elapsed_days / max(window_days, 1.0), 1.0)

        blended_daily = (
            elapsed_fraction         * observed_daily_rate
            + (1.0 - elapsed_fraction) * bayesian_daily_rate
        )

        wp  = float(max(count_in_window,
                        round(count_in_window + blended_daily * remaining_days)))
        ws  = float(np.sqrt(
            (ps ** 2) * (remaining_days / max(window_days, 1.0))
        )) * (1.0 - elapsed_fraction * 0.5)

        wlo = float(max(count_in_window, round(wp - 1.96 * ws)))
        whi = float(round(wp + 1.96 * ws))

        return wp, wlo, whi

    except Exception as exc:
        logger.debug("_compute_window_pred error: %s", exc)
        return None, None, None


def log_bet_answer_probabilities(
    active_trackings: list[dict],
    prediction:       dict,
    posts_in_windows: dict[str, int],
) -> None:
    """
    Logs a formatted probability table (console only) for every active bet.
    Called every poll cycle whether or not new tweets were found.

    Layout per bet
    --------------
    📌 [1/2] <market title>
       ⏳ Time Remaining : 2d 4h 17m
       ✅ Tweets in Window: 143
       🔮 Predicted Total : 312  [95% CI: 278 – 346]  σ≈17.3

       Interval       Prob    Bar
       ─────────────────────────────────────────────────
       280-299       18.4%  █████████
       300-319       35.1%  █████████████████  ◄
       320-339       29.8%  ██████████████
       340-359       12.6%  ██████
       360-379        4.1%  ██
       ─────────────────────────────────────────────────
       * Probabilities are normalised over displayed intervals only.
    """
    if not active_trackings:
        logger.info("📊 No active bets — skipping interval probability log.")
        return

    now_str = (
        datetime.now(timezone.utc)
        .astimezone(EST_TZ)
        .strftime("%b %d, %Y %I:%M:%S %p EST")
    )

    logger.info("=" * 70)
    logger.info("📊  BET ANSWER INTERVAL PROBABILITIES — %s", now_str)
    logger.info("=" * 70)

    for idx, tracking in enumerate(active_trackings, start=1):
        t_id    = tracking.get("id", "")
        t_title = tracking.get("title", f"Market #{idx}")
        end_r   = tracking.get("endDate", "")

        # ── Time remaining ─────────────────────────────────────────────────
        time_left_str = "N/A"
        try:
            end_dt_utc = pd.to_datetime(end_r, utc=True).to_pydatetime()
            delta_secs = (
                end_dt_utc - datetime.now(timezone.utc)
            ).total_seconds()
            d = int(max(delta_secs / 86_400, 0))
            h = int(max((delta_secs % 86_400) / 3_600, 0))
            m = int(max((delta_secs % 3_600)  / 60,    0))
            time_left_str = f"{d}d {h}h {m}m"
        except Exception:
            pass

        count_in_window = posts_in_windows.get(t_id, 0)

        # ── Window prediction ──────────────────────────────────────────────
        wp, wlo, whi = _compute_window_pred(
            tracking        = tracking,
            prediction      = prediction,
            count_in_window = count_in_window,
        )

        logger.info("")
        logger.info(
            "  📌 [%d/%d]  %s", idx, len(active_trackings), t_title
        )
        logger.info("     ⏳ Time Remaining  : %s",    time_left_str)
        logger.info("     ✅ Tweets in Window: %d",    count_in_window)

        if wp is None or wlo is None or whi is None:
            logger.info(
                "     ⚠️  Cannot compute window prediction — insufficient data."
            )
            logger.info("  " + "-" * 68)
            continue

        # σ derived from 95% CI  →  (upper - lower) / (2 × 1.96)
        std = max((whi - wlo) / 3.92, 1.0)

        logger.info(
            "     🔮 Predicted Total : %d  [95%% CI: %d – %d]  σ≈%.1f",
            int(wp), int(wlo), int(whi), std,
        )

        # ── Interval probabilities ─────────────────────────────────────────
        intervals = compute_interval_probabilities(
            mean     = wp,
            std      = std,
            ci_lower = wlo,
            ci_upper = whi,
        )

        if not intervals:
            logger.info(
                "     ⚠️  No defined intervals overlap with CI [%d – %d].",
                int(wlo), int(whi),
            )
            logger.info("  " + "-" * 68)
            continue

        logger.info("")
        logger.info("     %-12s  %8s  %s", "Interval", "Prob", "Bar")
        logger.info("     " + "─" * 55)

        for item in intervals:
            pct     = item["prob_pct"]
            bar_len = int(round(pct / 2))     # 50 chars == 100 %
            bar     = "█" * bar_len
            # Arrow marks the interval that contains the point prediction
            marker  = "  ◄" if item["low"] <= wp <= item["high"] else ""
            logger.info(
                "     %-12s  %6.1f%%  %s%s",
                item["label"], pct, bar, marker,
            )

        logger.info("     " + "─" * 55)
        logger.info(
            "     * Probabilities are normalised over displayed intervals only."
        )
        logger.info("  " + "-" * 68)

    logger.info("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# DISCORD NOTIFIER
# ──────────────────────────────────────────────────────────────────────────────

class DiscordNotifier:
    def __init__(self, session: aiohttp.ClientSession):
        self._session     = session
        self._webhook_url = DISCORD_WEBHOOK_URL
        self._enabled     = True

    async def send(
        self,
        *,
        new_tweet_count:  int,
        cumulative_count: int,
        tweets_this_week: int,
        prediction:       dict,
        active_trackings: list[dict],
        posts_in_windows: dict[str, int],
    ) -> None:
        if not self._enabled:
            logger.info("Discord disabled — skipping.")
            return

        predicted = prediction["predicted_weekly_total"]
        bayesian  = prediction["bayesian_weekly_total"]
        ci_low    = prediction["adjusted_ci_95_lower"]
        ci_high   = prediction["adjusted_ci_95_upper"]
        ef        = prediction["event_factor"]
        ef_pct    = prediction["event_adjustment_pct"]
        days_obs  = prediction["days_observed"]
        days_rem  = prediction["days_remaining"]
        ef_sign   = "+" if ef >= 0 else ""

        avg_per_day    = tweets_this_week / max(days_obs, 1)
        pace_projected = round(tweets_this_week + avg_per_day * days_rem)

        colour = (
            0x2ECC71 if ef >= 0.05
            else 0xE74C3C if ef <= -0.05
            else 0x3498DB
        )

        now_est = datetime.now(timezone.utc).astimezone(EST_TZ).strftime(
            "%b %d, %Y %I:%M:%S %p EST"
        )

        # ── Embed 1: Core stats (always sent) ─────────────────────────────────
        core_fields: list[dict] = [
            {
                "name":   "🕐 Detected At",
                "value":  now_est,
                "inline": False,
            },
            {
                "name":   "🆕 New Tweets This Poll",
                "value":  f"**+{new_tweet_count}**",
                "inline": True,
            },
            {
                "name":   "📊 Cumulative All-Time",
                "value":  f"**{cumulative_count:,}**",
                "inline": True,
            },
            {
                "name":   "📅 Tweets This Week (Mon–Sun)",
                "value":  f"**{tweets_this_week:,}**",
                "inline": True,
            },
            {
                "name":   "\u200b",
                "value":  "\u200b",
                "inline": False,
            },
            {
                "name":   "🔮 Predicted Weekly Total",
                "value":  f"**{predicted:,}** tweets",
                "inline": True,
            },
            {
                "name":   "📐 95% Confidence Interval",
                "value":  f"{ci_low:,} – {ci_high:,}",
                "inline": True,
            },
            {
                "name":   "🧮 Bayesian Estimate (pre-EF)",
                "value":  f"{bayesian:,}",
                "inline": True,
            },
            {
                "name":   "📰 News Event Factor",
                "value":  f"{ef_sign}{ef:.4f}  ({ef_sign}{ef_pct:.1f}%)",
                "inline": True,
            },
            {
                "name":   "📈 Linear Pace Projection",
                "value":  f"{pace_projected:,}",
                "inline": True,
            },
            {
                "name":   "📆 Days Observed / Remaining",
                "value":  f"{days_obs} observed · {days_rem} remaining",
                "inline": True,
            },
        ]

        embeds: list[dict] = [
            {
                "title":     f"🐦 Elon Tweeted  (+{new_tweet_count})",
                "color":     colour,
                "fields":    core_fields,
                "footer":    {
                    "text": (
                        "Elon Tweet Predictor · "
                        "Bayesian + News Factor Model · "
                        "xtracker.polymarket.com"
                    )
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]

        # ── Embed 2…N: One embed per active tracking period ───────────────────
        if active_trackings:
            for idx, tracking in enumerate(active_trackings, start=1):
                t_id    = tracking.get("id", "")
                t_title = tracking.get("title", f"Market #{idx}")
                target  = tracking.get("target")
                start_r = tracking.get("startDate", "")
                end_r   = tracking.get("endDate", "")

                # ── Window date string and duration ──
                window_str  = "N/A"
                window_days = 7.0
                if start_r and end_r:
                    try:
                        s = (
                            pd.to_datetime(start_r, utc=True)
                            .tz_convert(EST_TZ)
                            .strftime("%b %d %I:%M %p")
                        )
                        e = (
                            pd.to_datetime(end_r, utc=True)
                            .tz_convert(EST_TZ)
                            .strftime("%b %d %I:%M %p EST")
                        )
                        window_str  = f"{s} → {e}"
                        start_dt    = pd.to_datetime(start_r, utc=True).to_pydatetime()
                        end_dt      = pd.to_datetime(end_r,   utc=True).to_pydatetime()
                        window_days = max(
                            (end_dt - start_dt).total_seconds() / 86_400, 1.0
                        )
                    except Exception:
                        pass

                # ── Time remaining in market ──
                time_left_str = "N/A"
                try:
                    end_dt_utc    = pd.to_datetime(end_r, utc=True).to_pydatetime()
                    delta_secs    = (end_dt_utc - datetime.now(timezone.utc)).total_seconds()
                    days_left     = max(delta_secs / 86_400, 0.0)
                    hours_left    = max((delta_secs % 86_400) / 3600, 0.0)
                    time_left_str = f"{int(days_left)}d {int(hours_left)}h remaining"
                except Exception:
                    pass

                count_in_window = posts_in_windows.get(t_id, 0)

                # ── Window-scaled Bayesian prediction (anchored to observed) ──
                window_pred_str = "N/A"
                wp = wlo = whi = None
                try:
                    pm  = prediction["posterior_mean"]
                    ps  = prediction["posterior_std"]
                    efm = 1.0 + ef

                    # ── How much of the window has elapsed vs. remains ──
                    now_utc        = datetime.now(timezone.utc)
                    start_dt_utc   = pd.to_datetime(start_r, utc=True).to_pydatetime()
                    end_dt_utc     = pd.to_datetime(end_r,   utc=True).to_pydatetime()

                    elapsed_days   = max(
                        (now_utc - start_dt_utc).total_seconds() / 86_400, 0.0
                    )
                    remaining_days = max(
                        (end_dt_utc - now_utc).total_seconds() / 86_400, 0.0
                    )
                    # Clamp elapsed so it never exceeds total window
                    elapsed_days   = min(elapsed_days, window_days)

                    # ── Observed daily rate from tweets already counted ──
                    # Use a small floor (0.5) to avoid division by zero at
                    # the very start of a window
                    observed_daily_rate = count_in_window / max(elapsed_days, 0.5)

                    # ── Bayesian prior daily rate (from weekly model) ──
                    bayesian_daily_rate = (pm / 7.0) * efm

                    # ── Blend: weight observed rate by fraction of window elapsed.
                    # At 0% elapsed  → trust Bayesian fully.
                    # At 100% elapsed → trust observed rate fully.
                    # This ensures early in the window we don't over-anchor on
                    # a small sample, but late in the window we respect reality.
                    elapsed_fraction = min(elapsed_days / max(window_days, 1.0), 1.0)
                    blended_daily    = (
                        elapsed_fraction       * observed_daily_rate +
                        (1.0 - elapsed_fraction) * bayesian_daily_rate
                    )

                    # ── Final prediction:
                    # Already observed + blended rate projected over remaining days
                    projected_remaining = blended_daily * remaining_days
                    wp  = max(count_in_window, round(
                        count_in_window + projected_remaining
                    ))

                    # ── Uncertainty: shrinks as window progresses because
                    # remaining_days shrinks, so the CI tightens naturally ──
                    ws  = float(np.sqrt(
                        (ps ** 2) * (remaining_days / max(window_days, 1.0))
                    ))
                    # Also scale uncertainty down by elapsed_fraction so CI
                    # reflects how much has already been locked in by observation
                    ws  = ws * (1.0 - elapsed_fraction * 0.5)

                    wlo = max(count_in_window, round(
                        wp - 1.96 * ws
                    ))
                    whi = round(wp + 1.96 * ws)

                    # ── Hawkes-conditional override (when engine fitted) ──
                    # Replaces the blended-rate point forecast and CI with
                    # the self-exciting conditional expectation (burst-
                    # amplified / silence-suppressed); the pace label below
                    # still reports the raw observed rate.
                    if FORECAST_ENGINE is not None and FORECAST_ENGINE.is_fitted:
                        hawkes_fc = FORECAST_ENGINE.forecast_window(
                            count_in_window = count_in_window,
                            start_dt        = start_dt_utc,
                            end_dt          = end_dt_utc,
                            now             = now_utc,
                        )
                        if hawkes_fc is not None:
                            wp  = int(hawkes_fc[0])
                            wlo = int(hawkes_fc[1])
                            whi = int(hawkes_fc[2])

                    # ── Human-readable elapsed/remaining label ──
                    el_d = int(elapsed_days)
                    el_h = int((elapsed_days - el_d) * 24)
                    re_d = int(remaining_days)
                    re_h = int((remaining_days - re_d) * 24)
                    pace_label = (
                        f"elapsed {el_d}d {el_h}h · "
                        f"{re_d}d {re_h}h left · "
                        f"observed rate {observed_daily_rate:.1f}/day"
                    )

                    window_pred_str = (
                        f"**{wp:,}**  [CI: {wlo:,}–{whi:,}]\n"
                        f"_{pace_label}_"
                    )
                except Exception as exc:
                    logger.debug("Window pred error %s: %s", t_id, exc)

                # ── Progress bar vs target ──
                if target and int(target) > 0:
                    pct    = min(
                        round(count_in_window / int(target) * 100, 1), 100.0
                    )
                    filled = int(pct / 10)
                    bar    = "🟩" * filled + "⬜" * (10 - filled)
                    progress_str = (
                        f"{bar}  {pct}%  ({count_in_window:,}/{int(target):,})"
                    )
                    target_str = f"{int(target):,} tweets"
                else:
                    progress_str = (
                        f"{count_in_window:,} tweets so far (no target set)"
                    )
                    target_str = "—"

                # ── Optimal bet recommendation ──
                bet_str = "N/A — no target or CI available"
                if target and wp is not None and wlo is not None and whi is not None:
                    try:
                        # Attempt to read market YES price from tracking metadata.
                        # Polymarket may expose this under several field names.
                        raw_price = (
                            tracking.get("yesPrice")          or
                            tracking.get("yes_price")         or
                            tracking.get("probability")       or
                            (tracking.get("outcomePrices") or [None])[0]
                        )
                        market_yes = float(raw_price) if raw_price is not None else 0.50
                        # Normalise: some endpoints return 0–100 scale
                        if market_yes > 1.0:
                            market_yes /= 100.0
                        market_yes = float(np.clip(market_yes, 0.01, 0.99))

                        bet_info = calculate_optimal_bet(
                            predicted_total  = wp,
                            ci_lower         = wlo,
                            ci_upper         = whi,
                            target           = int(target),
                            market_yes_price = market_yes,
                            kelly_fraction   = 0.25,   # quarter-Kelly = conservative
                            bankroll         = 100.0,  # size relative to \$100 bankroll
                        )
                        bet_str = bet_info["recommendation"]
                    except Exception as exc:
                        logger.debug("Bet calc error %s: %s", t_id, exc)
                        bet_str = "⚠️ Could not compute — check market price field"

                # ── Build this market's embed fields ──
                market_fields: list[dict] = [
                    {
                        "name":   "🗓️ Window",
                        "value":  window_str,
                        "inline": False,
                    },
                    {
                        "name":   "⏳ Time Remaining",
                        "value":  time_left_str,
                        "inline": True,
                    },
                    {
                        "name":   "🎯 Target",
                        "value":  target_str,
                        "inline": True,
                    },
                    {
                        "name":   "✅ Tweets in Window",
                        "value":  f"**{count_in_window:,}**",
                        "inline": True,
                    },
                    {
                        "name":   "🔮 Window Prediction",
                        "value":  window_pred_str,
                        "inline": False,
                    },
                    {
                        "name":   "📊 Progress",
                        "value":  progress_str,
                        "inline": False,
                    },
                    {
                        "name":   "💰 Optimal Bet Suggestion",
                        "value":  bet_str,
                        "inline": False,
                    },
                ]

                embeds.append({
                    "title":  f"📌 [{idx}/{len(active_trackings)}] {t_title}",
                    "color":  colour,
                    "fields": market_fields[:25],   # Discord hard limit per embed
                })

        else:
            # No active markets — append a note to the core embed
            embeds[0]["fields"].append({
                "name":   "📌 Active Tracking Periods",
                "value":  "None currently active",
                "inline": False,
            })

        # ── Post embeds in batches of 10 (Discord webhook limit per request) ──
        DISCORD_EMBED_LIMIT = 10

        async def _post_payload(embed_batch: list[dict]) -> None:
            payload = {"embeds": embed_batch}
            try:
                async with self._session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 204):
                        logger.info(
                            "Discord batch sent ✅ (%d embed(s))",
                            len(embed_batch),
                        )
                    else:
                        body = await resp.text()
                        logger.warning(
                            "Discord webhook returned %d: %s",
                            resp.status, body[:300],
                        )
            except asyncio.TimeoutError:
                logger.warning("Discord webhook timed out.")
            except Exception as exc:
                logger.error("Discord webhook error: %s", exc)

        for i in range(0, len(embeds), DISCORD_EMBED_LIMIT):
            batch = embeds[i : i + DISCORD_EMBED_LIMIT]
            await _post_payload(batch)
            if i + DISCORD_EMBED_LIMIT < len(embeds):
                # Brief pause between batches to avoid Discord rate limits
                await asyncio.sleep(1.0)

   
# ──────────────────────────────────────────────────────────────────────────────
# XTRACKER REST CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class XTrackerClient:
    def __init__(self, session: aiohttp.ClientSession):
        self._session = session

    async def _get(
        self, path: str, params: Optional[dict] = None
    ) -> dict | list:
        url = f"{XTRACKER_BASE_URL}{path}"
        async with self._session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_user(self) -> dict:
        data = await self._get(
            f"/users/{ELON_HANDLE}", {"platform": PLATFORM}
        )
        return data.get("data", data)

    async def get_posts(
        self,
        start_date: Optional[datetime] = None,
        end_date:   Optional[datetime] = None,
    ) -> list:
        params = {"platform": PLATFORM}
        if start_date:
            params["startDate"] = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end_date:
            params["endDate"]   = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        data = await self._get(f"/users/{ELON_HANDLE}/posts", params)
        return data.get("data", []) if isinstance(data, dict) else data

    async def get_all_trackings(self) -> list:
        data = await self._get(
            f"/users/{ELON_HANDLE}/trackings", {"platform": PLATFORM}
        )
        return data.get("data", []) if isinstance(data, dict) else data

    async def get_active_trackings(self) -> list[dict]:
        """
        Return ALL currently active tracking periods for Elon.
        Falls back to manual date filtering if activeOnly param fails.
        """
        try:
            data = await self._get(
                f"/users/{ELON_HANDLE}/trackings",
                {"platform": PLATFORM, "activeOnly": "true"},
            )
            trackings = data.get("data", []) if isinstance(data, dict) else data
            if trackings:
                return trackings
        except Exception as exc:
            logger.warning("activeOnly fetch failed, falling back: %s", exc)

        # Fallback: fetch all and filter by date manually
        try:
            all_trackings = await self.get_all_trackings()
            now    = datetime.now(timezone.utc)
            active = []
            for t in all_trackings:
                start_raw = t.get("startDate")
                end_raw   = t.get("endDate")
                if not start_raw or not end_raw:
                    continue
                start_dt = pd.to_datetime(start_raw, utc=True).to_pydatetime()
                end_dt   = pd.to_datetime(end_raw,   utc=True).to_pydatetime()
                if start_dt <= now <= end_dt:
                    active.append(t)
            return active
        except Exception as exc:
            logger.warning("Could not fetch active trackings: %s", exc)
            return []

    async def get_tracking_stats(self, tracking_id: str) -> dict:
        data = await self._get(
            f"/trackings/{tracking_id}", {"includeStats": "true"}
        )
        return data.get("data", {})

    async def get_total_count_for_range(
        self, start_date: datetime, end_date: datetime
    ) -> int:
        return len(await self.get_posts(start_date, end_date))

    def _tracking_fields(
        tracking:         dict,
        tweets_this_week: int,
        prediction:       dict,
    ) -> list[dict]:
        """
        Returns Discord embed field dicts for one tracking period,
        including a Bayesian predicted-vs-target assessment.
        """
        title     = tracking.get("title", "Unnamed Market")
        start_raw = tracking.get("startDate", "")
        end_raw   = tracking.get("endDate", "")
        target    = tracking.get("target")

        # Date window + time remaining
        window                   = "N/A"
        days_remaining_in_market = None
        if start_raw and end_raw:
            try:
                start_est = (
                    pd.to_datetime(start_raw, utc=True)
                    .tz_convert(EST_TZ)
                    .strftime("%b %d %I:%M %p")
                )
                end_dt  = pd.to_datetime(end_raw, utc=True)
                end_est = end_dt.tz_convert(EST_TZ).strftime("%b %d %I:%M %p EST")
                window  = f"{start_est} → {end_est}"

                now_utc = datetime.now(timezone.utc)
                delta   = (end_dt.to_pydatetime() - now_utc).total_seconds()
                days_remaining_in_market = max(delta / 86400, 0)
            except Exception:
                pass

        # Progress bar vs target
        progress_str        = "N/A"
        predicted_vs_target = "N/A"
        if target:
            try:
                target_int = int(target)
                pct        = min(round(tweets_this_week / target_int * 100, 1), 100)
                filled     = int(pct / 10)
                bar        = "🟩" * filled + "⬜" * (10 - filled)
                progress_str = f"{bar} {pct}%  ({tweets_this_week}/{target_int})"

                predicted = prediction["predicted_weekly_total"]
                ci_low    = prediction["adjusted_ci_95_lower"]
                ci_high   = prediction["adjusted_ci_95_upper"]

                if ci_low > target_int:
                    verdict = "✅ Likely to **EXCEED** target"
                elif ci_high < target_int:
                    verdict = "❌ Likely to **MISS** target"
                elif predicted > target_int:
                    verdict = "🟡 Predicted to **exceed** (but CI overlaps)"
                else:
                    verdict = "🟡 Predicted to **miss** (but CI overlaps)"

                predicted_vs_target = (
                f"{verdict}\n"
                f"Predicted: **{predicted:,}**  "
                f"[CI: {ci_low:,}–{ci_high:,}]  "
                f"vs Target: **{target_int:,}**"
                )
            except Exception:
                pass

        time_left = (
            f"{days_remaining_in_market:.1f} days remaining"
            if days_remaining_in_market is not None
            else "N/A"
    )   

        fields = [
            {
                "name":  f"🏷️  {title}",
                "value": (
                    f"**Window:** {window}\n"
                    f"**Time left:** {time_left}\n"
                    f"**Target:** {target if target else 'N/A'}\n"
                    f"**Progress:** {progress_str}"
                ),
                "inline": False,
            },
            {
                "name":   "🎯 Bayesian Assessment",
                "value":  predicted_vs_target,
                "inline": False,
            },
            {
                "name":   "\u200b",
                "value":  "─────────────────",
                "inline": False,
            },
        ]
        return fields


# ──────────────────────────────────────────────────────────────────────────────
# TWEET PROCESSOR
# ──────────────────────────────────────────────────────────────────────────────

class TweetProcessor:
    def __init__(
        self,
        csv_manager:        CSVManager,
        model:              BayesianTweetForecaster,
        pattern_analyzer:   TemporalPatternAnalyzer,
        news_scanner:       NewsScanner,
        event_analyzer:     EventFactorAnalyzer,
        deviation_detector: DeviationDetector,
        discord_notifier:   DiscordNotifier,
        xtracker_client:    XTrackerClient,
    ):
        self.csv            = csv_manager
        self.model          = model
        self.patterns       = pattern_analyzer
        self.news           = news_scanner
        self.event_analyzer = event_analyzer
        self.deviation      = deviation_detector
        self.discord        = discord_notifier
        self.client         = xtracker_client

        self._daily_counts:          dict[int, int]     = defaultdict(int)
        self._current_week_key:      Optional[str]      = None
        self._last_known_cumulative: int                = (
            csv_manager.latest_cumulative_count()
        )
        self._seen_tracking_ids:     set[str]           = set()
        self._news_scan_lock:        asyncio.Lock       = asyncio.Lock()
        self._active_trackings:      list[dict]         = []
        self._tracking_cache_ts:     Optional[datetime] = None
        self._tracking_cache_ttl:    int                = 300
        # Stores the most-recent Bayesian prediction dict so the poll loop
        # can log interval probabilities even when no new tweets arrive.
        self._latest_prediction: Optional[dict] = None

        logger.info(
            "TweetProcessor ready. Cumulative tweets in CSV: %d",
            self._last_known_cumulative,
        )

    async def _get_active_trackings(self) -> list[dict]:
        """Return cached active trackings list; refresh if stale."""
        now = datetime.now(timezone.utc)
        if (
            not self._active_trackings or
            self._tracking_cache_ts is None or
            (now - self._tracking_cache_ts).total_seconds() > self._tracking_cache_ttl
        ):
            self._active_trackings  = await self.client.get_active_trackings()
            self._tracking_cache_ts = now
            logger.info(
                "Active trackings refreshed: %d market(s) found.",
                len(self._active_trackings),
            )
        return self._active_trackings

    async def ingest_posts(self, posts: list[dict]) -> int:
        if not posts:
            return 0

        last_ts   = self.csv.latest_timestamp()
        new_posts = []

        for post in posts:
            raw_ts = (
                post.get("createdAt") or
                post.get("timestamp") or
                post.get("created_at")
            )
            if not raw_ts:
                continue
            ts = pd.to_datetime(raw_ts, utc=True).to_pydatetime()
            if last_ts is None or ts > last_ts:
                new_posts.append((ts, post))

        if not new_posts:
            return 0

        new_posts.sort(key=lambda x: x[0])

        active_trackings = await self._get_active_trackings()

        for ts, _ in new_posts:
            self._last_known_cumulative += 1
            self.csv.append_row(ts, self._last_known_cumulative)

            est_ts   = ts.astimezone(EST_TZ)
            dow      = est_ts.weekday()
            week_key = est_ts.strftime("%Y-W%W")

            if week_key != self._current_week_key:
                logger.info(
                    "New week detected: %s → resetting daily counts & posterior.",
                    week_key,
                )
                self._daily_counts     = defaultdict(int)
                self._current_week_key = week_key
                self.model._reset_week_state()

            self._daily_counts[dow] += 1
            tweets_this_week = sum(self._daily_counts.values())
            prediction       = self.model.update(dow, self._daily_counts[dow])
            self._latest_prediction = prediction

            deviation_report = self.deviation.evaluate(
                day_of_week    = dow,
                actual_count   = self._daily_counts[dow],
                posterior_mean = prediction["posterior_mean"],
                day_weights    = self.model.day_weights,
            )

            if deviation_report["is_high_deviation"]:
                asyncio.create_task(
                    self._run_news_scan(
                        trigger = (
                            f"deviation_"
                            f"{deviation_report['deviation_direction']}"
                        ),
                        z_score = deviation_report["z_score"],
                    )
                )

        # Refit the Hawkes/conditional engine on the updated history.
        # Runs in a worker thread (the MLE can take a few seconds) and is
        # throttled internally; diagnostics are DEBUG-level only.
        if FORECAST_ENGINE is not None:
            try:
                asyncio.create_task(asyncio.to_thread(FORECAST_ENGINE.refit))
            except Exception as exc:
                logger.debug("Hawkes refit scheduling error: %s", exc)

    # fetch per-window counts then send Discord notification
        posts_in_windows: dict[str, int] = {}
        if active_trackings:
            try:
                for t in active_trackings:
                    t_id    = t.get("id", "")
                    start_r = t.get("startDate")
                    end_r   = t.get("endDate")
                    if not t_id or not start_r or not end_r:
                        continue
                    try:
                        stats = await self.client.get_tracking_stats(t_id)
                        total = stats.get("stats", {}).get("total")
                        if total is not None:
                            posts_in_windows[t_id] = int(total)
                            continue
                    except Exception:
                        pass
                    try:
                        start_dt = pd.to_datetime(start_r, utc=True).to_pydatetime()
                        end_dt   = pd.to_datetime(end_r,   utc=True).to_pydatetime()
                        posts_in_windows[t_id] = (
                            await self.client.get_total_count_for_range(
                                start_dt, end_dt
                            )
                        )
                    except Exception as exc:
                        logger.warning("Window count error %s: %s", t_id, exc)
                        posts_in_windows[t_id] = 0
            except Exception as exc:
                logger.warning("Could not build posts_in_windows: %s", exc)

        logger.info("Attempting to send discord notification...")
        await self.discord.send(
        new_tweet_count  = 1,
        cumulative_count = self._last_known_cumulative,
        tweets_this_week = tweets_this_week,
        prediction       = prediction,
        active_trackings = active_trackings,
        posts_in_windows = posts_in_windows,
    )   

        self.model.save()

        logger.info(
            "Ingested %d new tweet(s). Cumulative: %d",
            len(new_posts), self._last_known_cumulative,
        )
        return len(new_posts)

    async def get_current_prediction(self) -> dict:
        """
        Returns the most recent Bayesian prediction dict.

        If no tweets have been ingested yet this session we call model.update()
        with today's current daily count (which may be 0) so the poll loop
        always has a valid prediction to work with.
        """
        if self._latest_prediction is not None:
            return self._latest_prediction

        now_est  = datetime.now(timezone.utc).astimezone(EST_TZ)
        dow      = now_est.weekday()
        count    = self._daily_counts.get(dow, 0)
        pred     = self.model.update(dow, count)
        self._latest_prediction = pred
        return pred

    async def fetch_window_counts(
        self, active_trackings: list[dict]
    ) -> dict[str, int]:
        """
        Fetches the tweet count for every active tracking window.
        Tries the /trackings/{id} stats endpoint first; falls back to
        counting posts manually if that fails.
        Extracted here so both ingest_posts and poll_loop can share it.
        """
        posts_in_windows: dict[str, int] = {}

        for t in active_trackings:
            t_id    = t.get("id", "")
            start_r = t.get("startDate")
            end_r   = t.get("endDate")
            if not t_id or not start_r or not end_r:
                continue

            # Attempt stats endpoint first (faster, single request)
            try:
                stats = await self.client.get_tracking_stats(t_id)
                total = stats.get("stats", {}).get("total")
                if total is not None:
                    posts_in_windows[t_id] = int(total)
                    continue
            except Exception:
                pass

            # Fallback: count posts in date range
            try:
                start_dt = pd.to_datetime(start_r, utc=True).to_pydatetime()
                end_dt   = pd.to_datetime(end_r,   utc=True).to_pydatetime()
                posts_in_windows[t_id] = (
                    await self.client.get_total_count_for_range(
                        start_dt, end_dt
                    )
                )
            except Exception as exc:
                logger.warning(
                    "fetch_window_counts fallback error %s: %s", t_id, exc
                )
                posts_in_windows[t_id] = 0

        return posts_in_windows

    async def _run_news_scan(
        self,
        trigger:  str   = "periodic",
        z_score:  float = 0.0,
        lookback: int   = NEWS_LOOKBACK_HOURS,
    ) -> None:
        async with self._news_scan_lock:
            event_logger.info(
                "News scan triggered  source=%-30s  z_score=%+.2f",
                trigger, z_score,
            )
            articles = await self.news.fetch(lookback_hours=lookback)
            analysis = self.event_analyzer.analyze(articles)
            self.model.event_tracker.update(
                analysis["event_factor"], source=trigger
            )
            self.model.save()
            event_logger.info(
                "Post-scan: %s", self.model.event_tracker.summary()
            )

    async def check_and_retrain(self) -> None:
        try:
            trackings = await self.client.get_all_trackings()
        except Exception as exc:
            logger.warning("Could not fetch trackings: %s", exc)
            return

        now_utc = datetime.now(timezone.utc)

        for tracking in trackings:
            tracking_id = tracking.get("id", "")
            if not tracking_id or tracking_id in self._seen_tracking_ids:
                continue

            end_raw = tracking.get("endDate")
            if not end_raw:
                continue

            end_dt    = pd.to_datetime(end_raw, utc=True).to_pydatetime()
            is_active = tracking.get("isActive", True)

            if is_active or end_dt > now_utc:
                continue

            logger.info(
                "Resolved tracking: %s  (ended %s)",
                tracking_id, end_dt.isoformat(),
            )

            try:
                stats = await self.client.get_tracking_stats(tracking_id)
                total = stats.get("stats", {}).get("total")

                if total is None:
                    start_dt = pd.to_datetime(
                        tracking.get("startDate"), utc=True
                    ).to_pydatetime()
                    total = await self.client.get_total_count_for_range(
                        start_dt, end_dt
                    )

                logger.info(
                    "Official total for %s: %d", tracking_id, total
                )
                new_weights = self.patterns.analyze_and_log()
                self.model.retrain(int(total), new_weights)
                self.model.save()
                self._seen_tracking_ids.add(tracking_id)
                self._active_trackings  = []
                self._tracking_cache_ts = None

            except Exception as exc:
                logger.error(
                    "Retraining failed (%s): %s", tracking_id, exc
                )


# ──────────────────────────────────────────────────────────────────────────────
# ASYNC TASK LOOPS
# ──────────────────────────────────────────────────────────────────────────────

async def poll_loop(processor: TweetProcessor) -> None:
    """
    Polls XTracker REST API every POLL_INTERVAL_SEC (5 minutes).
    Fetches all posts since the last recorded timestamp and ingests them.

    After every poll — whether or not new tweets were found — logs the
    Bayesian interval probabilities for each active bet to the console.
    """
    logger.info(
        "REST polling loop started — interval: %ds.", POLL_INTERVAL_SEC
    )

    while True:
        # ── 1. Fetch & ingest new tweets ───────────────────────────────────
        try:
            since = processor.csv.latest_timestamp() or (
                datetime.now(timezone.utc) - timedelta(minutes=10)
            )
            posts = await processor.client.get_posts(start_date=since)
            new_n = await processor.ingest_posts(posts)
            if new_n:
                logger.info("Poll: ingested %d new tweet(s).", new_n)
            else:
                logger.info("Poll: no new tweets.")
        except Exception as exc:
            logger.error("Poll error: %s", exc)

        # ── 2. Log interval probabilities for every active bet ─────────────
        # This runs unconditionally so the table updates every 5 minutes
        # even during quiet periods, reflecting the shrinking time-remaining.
        try:
            active_trackings = await processor._get_active_trackings()

            if active_trackings:
                prediction       = await processor.get_current_prediction()
                posts_in_windows = await processor.fetch_window_counts(
                    active_trackings
                )
                log_bet_answer_probabilities(
                    active_trackings = active_trackings,
                    prediction       = prediction,
                    posts_in_windows = posts_in_windows,
                )
            else:
                logger.info(
                    "📊 No active bets — skipping interval probability log."
                )

        except Exception as exc:
            logger.error(
                "Interval probability logging error: %s", exc
            )

        # ── 3. Sleep until next poll ───────────────────────────────────────
        await asyncio.sleep(POLL_INTERVAL_SEC)


async def news_scan_loop(processor: TweetProcessor) -> None:
    """Periodic news scan with EventFactor decay applied before each cycle."""
    logger.info(
        "News scan loop started — interval: %ds.", NEWS_SCAN_INTERVAL
    )
    while True:
        await asyncio.sleep(NEWS_SCAN_INTERVAL)
        processor.model.event_tracker.decay()
        await processor._run_news_scan(trigger="periodic_scheduled")


async def bet_resolution_loop(processor: TweetProcessor) -> None:
    """Checks for resolved tracking periods and retrains the model."""
    logger.info(
        "Bet-resolution monitor started — interval: %ds.", BET_CHECK_INTERVAL
    )
    while True:
        await asyncio.sleep(BET_CHECK_INTERVAL)
        await processor.check_and_retrain()


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BOOTSTRAP
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_model(
    csv_manager:      CSVManager,
    pattern_analyzer: TemporalPatternAnalyzer,
) -> BayesianTweetForecaster:

    if Path(MODEL_FILE).exists():
        try:
            model = BayesianTweetForecaster.load(MODEL_FILE)
            logger.info("Resuming from saved model.")
            return model
        except Exception as exc:
            logger.warning(
                "Could not load model (%s) — rebuilding.", exc
            )

    logger.info("Building fresh model from CSV…")
    df = csv_manager.load_dataframe()

    if df.empty:
        logger.info("No CSV data — using default prior (mean=100, std=40).")
        model = BayesianTweetForecaster(prior_mean=100.0, prior_std=40.0)
        model.save()
        return model

    df["DateTime_EST"] = df["DateTime_UTC"].dt.tz_convert(EST_TZ)
    df["DailyCount"]   = (
        df["Cumulative_Tweet_Count"].diff()
        .fillna(df["Cumulative_Tweet_Count"].iloc[0])
        .clip(lower=0)
    )
    df["YearWeek"]  = df["DateTime_EST"].dt.strftime("%Y-W%W")
    weekly_totals   = df.groupby("YearWeek")["DailyCount"].sum()

    prior_mean = (
        float(weekly_totals.mean()) if len(weekly_totals) > 0 else 100.0
    )
    prior_std = max(
        float(weekly_totals.std()) if len(weekly_totals) > 1 else 40.0,
        10.0,
    )
    day_weights = pattern_analyzer.analyze_and_log()

    model = BayesianTweetForecaster(
        prior_mean     = prior_mean,
        prior_std      = prior_std,
        day_weights    = day_weights,
        training_weeks = list(map(int, weekly_totals.tolist())),
    )
    logger.info(
        "Fresh model: %d historical weeks, prior mean=%.1f std=%.1f",
        len(weekly_totals), prior_mean, prior_std,
    )
    model.save()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ──────────────────────────────────────────────────────────────────────────────

def install_shutdown_handler(
    model: BayesianTweetForecaster,
    loop:  asyncio.AbstractEventLoop,
) -> None:
    def _shutdown(sig_name: str) -> None:
        logger.info("Received %s — saving model before exit…", sig_name)
        model.save()
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(
                sig, lambda s=sig.name: _shutdown(s)
            )
        except NotImplementedError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("=" * 70)
    logger.info(" Elon Musk Weekly Tweet Predictor — Bayesian + News + Discord")
    logger.info("=" * 70)

    csv_manager        = CSVManager(CSV_FILE)
    pattern_analyzer   = TemporalPatternAnalyzer(csv_manager)
    model              = bootstrap_model(csv_manager, pattern_analyzer)
    deviation_detector = DeviationDetector(csv_manager)
    event_analyzer     = EventFactorAnalyzer()

    # ── Hawkes / conditional-probability engine ──
    global FORECAST_ENGINE
    FORECAST_ENGINE = ConditionalForecastEngine(csv_manager)
    try:
        await asyncio.to_thread(FORECAST_ENGINE.refit, True)
    except Exception as exc:
        logger.debug("Initial Hawkes fit failed: %s", exc)
    model.forecast_engine = FORECAST_ENGINE

    connector = aiohttp.TCPConnector(limit=10)
    timeout   = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        xtracker_client  = XTrackerClient(session)
        news_scanner     = NewsScanner(session)
        discord_notifier = DiscordNotifier(session)

        processor = TweetProcessor(
            csv_manager        = csv_manager,
            model              = model,
            pattern_analyzer   = pattern_analyzer,
            news_scanner       = news_scanner,
            event_analyzer     = event_analyzer,
            deviation_detector = deviation_detector,
            discord_notifier   = discord_notifier,
            xtracker_client    = xtracker_client,
        )

        try:
            user_info = await xtracker_client.get_user()
            logger.info("Tracking: @%s  (%s)",
                        user_info.get("handle", ELON_HANDLE), PLATFORM)
        except Exception as exc:
            logger.warning("Could not fetch user info: %s", exc)

        # Initial backfill
        logger.info("Backfilling from last recorded timestamp…")
        since = csv_manager.latest_timestamp() or (
            datetime.now(timezone.utc) - timedelta(days=7)
        )
        try:
            posts    = await xtracker_client.get_posts(start_date=since)
            ingested = await processor.ingest_posts(posts)
            logger.info("Backfill complete: %d new tweets.", ingested)
        except Exception as exc:
            logger.warning("Backfill failed: %s", exc)

        # Refit the Hawkes engine on the freshly backfilled history
        try:
            await asyncio.to_thread(FORECAST_ENGINE.refit, True)
        except Exception as exc:
            logger.debug("Post-backfill Hawkes refit failed: %s", exc)

        # Seed EventFactor before going live
        logger.info("Running startup news scan…")
        await processor._run_news_scan(trigger="startup", lookback=24)

        # Check for already-resolved bets
        await processor.check_and_retrain()

        loop = asyncio.get_running_loop()
        install_shutdown_handler(model, loop)

        logger.info("All systems go — entering main loops.")
        try:
            await asyncio.gather(
                poll_loop(processor),
                bet_resolution_loop(processor),
                news_scan_loop(processor),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
