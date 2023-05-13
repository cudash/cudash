FROM python:3.11

COPY . /workdir

RUN cd /workdir && pip install -r requirements.txt && pip install matplotlib

WORKDIR /workdir
CMD ["bash", "-c", "python src/app.py"]

