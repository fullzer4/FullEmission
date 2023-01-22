FROM continuumio/miniconda3:latest

WORKDIR /api

COPY environment.yml ./
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "pytorchcpu", "/bin/bash", "-c"]

RUN source activate pytorchcpu

COPY . .

EXPOSE 5000:5000

CMD ["python", "index.py"]