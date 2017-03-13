FROM nvidia/cuda:7.5-cudnn4-devel

MAINTAINER Nikki Aldeborgh <nikki.aldeborgh@digitalglobe.com>

RUN apt-get -y update && apt-get -y \
    install python \
    build-essential \
    python-software-properties \
    software-properties-common \
    ipython \
    python-pip \
    python-scipy \
    python-numpy \
    python-dev \
    gdal-bin \
    python-gdal \
    libgdal-dev \
    libspatialite-dev \
    sqlite3 \
    libpq-dev \
    libcurl4-gnutls-dev \
    libproj-dev \
    libxml2-dev \
    libgeos-dev \
    libnetcdf-dev \
    libpoppler-dev \
    libspatialite-dev \
    libhdf4-alt-dev \
    libhdf5-serial-dev \
    wget \
    vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install Theano Keras h5py geojson sklearn gdal numpy geojsontools dataextractors

ADD ./bin /
COPY .theanorc /root/.theanorc
COPY keras.json /root/.keras/keras.json
