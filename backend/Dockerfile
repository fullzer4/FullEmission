FROM continuumio/miniconda3:latest

WORKDIR /api

COPY environment.yml ./
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "pytorchcpu", "/bin/bash", "-c"]
RUN conda init bash
RUN source activate pytorchcpu
RUN conda info -e
COPY . .

CMD ["python", "index.py"]

EXPOSE 5000:5000