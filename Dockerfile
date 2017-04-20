FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
  apt-get install -y --no-install-recommends software-properties-common && \
  add-apt-repository ppa:ubuntugis/ubuntugis-unstable && \
  apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y --no-install-recommends \
    apt-transport-https \
    build-essential \
    gdal-bin \
    git \
    libgdal-dev \
    lsb-release \
    python-dev \
    python-pip \
    python-setuptools \
    python-wheel \
    software-properties-common \
    wget && \
  apt-get clean

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -U numpy && \
  pip install -Ur requirements.txt && \
  pip install -U gevent gunicorn && \
  rm -rf /root/.cache

COPY . /app

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# override this accordingly; should be 2-4x $(nproc)
ENV WEB_CONCURRENCY 4
EXPOSE 8000
USER nobody

ENTRYPOINT ["gunicorn", "-k", "gevent", "-b", "0.0.0.0", "--timeout", "300", "--access-logfile", "-", "app:app"]
