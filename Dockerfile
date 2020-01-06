# AUTHOR: Jin Dong Yang
# EDITOR: Hubble SG
# DESCRIPTION: API Container for Face Detection Module

FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

LABEL maintainer='dongyang@hubble.sg'

ARG USER_HOME=/usr/local/spoofing_detection_api
ENV PYTHONPATH="$PYTHONPATH/:${USER_HOME}"
ENV PYTHONPATH="$PYTHONPATH/:${USER_HOME}/src"

EXPOSE 8000

RUN apt-get update -y \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && pip install -U pip setuptools wheel \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf \
      /var/lib/apt/lists/* \
      /tmp/* \
      /var/tmp/* \
      /usr/share/man \
      /usr/share/doc \
      /usr/share/doc-base

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

COPY ./entrypoint.sh /entrypoint.sh
COPY /config ${USER_HOME}/config
COPY /src ${USER_HOME}/src

WORKDIR ${USER_HOME}
ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "gunicorn" ]