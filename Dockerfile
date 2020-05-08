FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /
#COPY saved_data/ /

ADD ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt


ADD . /
#COPY . /

RUN apt-get -y install libsndfile1



CMD [ "python","main.py" ]


