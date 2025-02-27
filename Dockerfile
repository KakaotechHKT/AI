# Ubuntu 22.04 기반 이미지 사용
FROM ubuntu:22.04

# 필요한 패키지 설치 (Python 포함)
RUN apt update && apt install -y \
    python3 python3-pip python3-venv \
    curl wget git vim && \
    rm -rf /var/lib/apt/lists/*

# python3를 기본 python으로 설정
RUN ln -s /usr/bin/python3 /usr/bin/python

# Pip 최신 버전으로 업그레이드
RUN python -m pip install --upgrade pip

# 작업 디렉토리 생성 및 파일 복사
WORKDIR /app/babpat
COPY chatbot.py restaurants_def.csv requirements.txt .env main.py faiss_index.bin /app/babpat/

# requirements.txt 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 기본 명령어 설정 (bash 실행)
CMD ["bash"]
