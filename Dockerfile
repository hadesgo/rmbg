FROM ubuntu:22.04

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y wget

RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh \
    && bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p /anaconda \
    && rm Anaconda3-2023.03-1-Linux-x86_64.sh

ENV PATH=/anaconda/bin:$PATH

RUN conda update -y conda

COPY pytorch18.yml /app/

RUN conda env create -f pytorch18.yml

COPY . /app

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "pytorch18"]

CMD [ "python", "src/run.py" ]