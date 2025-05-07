# Dockerfile

# 1) 베이스 이미지 지정 (원하는 Python 버전으로 교체 가능)
FROM python:3.12-slim

# 2) 작업 디렉토리 생성/설정
WORKDIR /app

# 3) 캐시 활용을 위해 requirements.txt만 먼저 복사 후 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) 나머지 소스 코드 복사
COPY . .

# 5) Flask 실행을 위한 환경 변수 설정
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 6) 컨테이너 기동 시 실행 명령
#    - 개발용: flask run
#    - 운영용: gunicorn 권장
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]