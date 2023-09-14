FROM python:3.10-slim

# dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
        build-essential ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0

RUN pip3 install -U pip && pip3 install --upgrade pip

WORKDIR /solution

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY scorer.py .
COPY models/ ./models
COPY solution.py .
COPY models.py .

# input and output folders
RUN mkdir -p ./private/images && \
    mkdir -p ./private/labels && \
    mkdir -p ./output

# !!!! ONLY FOR THE TEST RUN - DELETE BEFORE SUBMITTING --->>>
#COPY images ./private/images
#COPY labels ./private/labels
# <<<---

CMD /bin/sh -c "python3 solution.py && python3 scorer.py"
