#!/bin/bash

if [ "$1" == "1" ]
then
    rm -rf samples/img
else
    mkdir -p samples/img/rgb
    mkdir -p samples/img/yuv

    ffmpeg -i samples/g01s20.avi -f image2 samples/img/rgb/img_%03d.jpg

    # RGB -> YCrCb
    for i in $(seq -f "%03g" 1 999)
    do
	convert samples/img/rgb/img_$i.jpg -colorspace YCbCr -separate samples/img/yuv/Yimg_$i.bmp
    done
    # Delete cr,cb images
    rm -rf samples/img/yuv/Yimg_*-[1-2].bmp
    # Convert gray images to gray.avi
    ffmpeg -i samples/img/yuv/'Yimg_%03d-0.bmp' -r 20 samples/gray.avi
    # Test
    source/bgslibrary/build/bgslibrary -uf -fn=samples/gray.avi
fi

# Cutting video into pieces with fixd duration
ffmpeg -i input.avi -ss 00:06:17 -t 00:07:30 -c copy output.avi
