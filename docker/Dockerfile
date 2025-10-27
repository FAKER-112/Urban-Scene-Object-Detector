FROM python:3.11-slim

WORKDIR /app
COPY . /app


RUN apt-get update && apt-get install -y libgl1 && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

 