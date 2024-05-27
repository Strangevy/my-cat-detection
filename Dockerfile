# 使用官方的Python运行时作为父镜像
FROM python:3.12-slim-buster

# 设置工作目录
WORKDIR /app

# 安装所需包和依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制项目文件到容器中
COPY . .

# 环境变量配置，确保在容器中也能正确读取
ENV MODEL_PATH /app/best.pt
ENV STREAM_URL http://192.168.233.160:4747/video
ENV TELEGRAM_BOT_TOKEN your_bot_token
ENV TELEGRAM_CHAT_ID your_chat_id

# 定义运行时命令
CMD ["python", "./app.py"]