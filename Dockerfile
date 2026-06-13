FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY elon_musk_tweet_predictorv6_hawkes_deployment_file_final.py .
RUN mkdir -p /app/data
ENV DATA_DIR=/app/data
STOPSIGNAL SIGTERM
CMD ["python", "elon_musk_tweet_predictorv6_hawkes_deployment_file_final.py"]
