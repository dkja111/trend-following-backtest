FROM python:3.10-slim

WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bt 패키지 직접 설치 - 최신 main 브랜치 사용
RUN pip install --no-cache-dir git+https://github.com/pmorissette/bt.git

# 나머지 소스 코드 복사
COPY . .

# 포트 노출
EXPOSE 8501

# 헬스체크 설정
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Streamlit 실행
ENTRYPOINT ["streamlit", "run", "trend01.py", "--server.port=8501", "--server.address=0.0.0.0"]
