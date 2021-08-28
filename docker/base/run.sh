#!/usr/bin/env bash

docker run --gpus all -ti --rm igibson/igibson:latest -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix
