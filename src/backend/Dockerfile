FROM python:3.8

RUN mkdir /api
WORKDIR /api

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "index.py"]