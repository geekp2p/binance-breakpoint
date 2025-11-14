
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     libglib2.0-0 libxext6 libsm6 libxrender1 libfreetype6 libpng16-16 tzdata  && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     MPLBACKEND=Agg     TZ=Asia/Bangkok

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "main.py", "--config", "config.yaml", "--out-dir", "out"]
