#!/bin/bash

if [ "$1" == "1" ]
then
    rm -rf samples/img
else
    mkdir -p samples/img
    ffmpeg -i samples/g01s20.avi -f image2 samples/img/img_%03d.jpg
fi
