#!/bin/bash

docker build -t wavenet/latest .
docker run -v $(pwd)/saved_data:/saved_data:rw --gpus all -it --rm --name wavenetbox wavenet/latest 

