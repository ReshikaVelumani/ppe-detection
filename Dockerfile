# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /ppe-detection

COPY requirements.txt requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

ENV PORT 5000

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
