FROM python:3.9.13

WORKDIR /api

RUN pip install flask flask_cors torch pandas

COPY . .

CMD ["python", "index.py"]

EXPOSE 5000:5000