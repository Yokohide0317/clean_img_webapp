FROM ubuntu:latest
USER root
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get -y install locales libgl1-mesa-dev && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9

RUN apt-get install -y vim less docker python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

WORKDIR /home/web
ADD requirements.txt .
ADD setup.sh .
RUN python3 -m pip install -r requirements.txt
RUN bash setup.sh

WORKDIR /home/web
CMD ["python3", "app.py"]
