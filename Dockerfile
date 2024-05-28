FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV MODEL_PATH /app/best.pt
ENV STREAM_URL http://192.168.233.160:4747/video
ENV TELEGRAM_BOT_TOKEN your_bot_token
ENV TELEGRAM_CHAT_ID your_chat_id

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "./app.py"]
