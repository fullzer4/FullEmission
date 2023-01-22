FROM python:3.8

WORKDIR /api

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "index.py"]