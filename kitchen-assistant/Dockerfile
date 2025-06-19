FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential

COPY requirements.txt .
RUN python3 -m pip install --upgrade --no-cache-dir -r requirements.txt
