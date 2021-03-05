# use existing anaconda3 image as base
FROM continuumio/anaconda3 
# install packages used in sh scripts
RUN apt-get update && apt-get install -y \
rsync \
make \
g++ \
texlive \
texlive-latex-extra \
dvipng \
texlive-science \
&& rm -rf /var/lib/apt/lists/*
# set workdir
WORKDIR /usr/src/ntw/network
# create env
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
