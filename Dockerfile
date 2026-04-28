FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY . .

RUN chmod +x entrypoint.sh 

ENTRYPOINT ["./entrypoint.sh"]
