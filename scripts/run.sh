#!/usr/bin/env bash

case "$1" in
    build)
        docker build --rm -t antispoofing_detection:latest .
        docker run -d -p 5000:8000 --name=antispoof antispoofing_detection
        ;;
    kill)
        docker stop antispoof
        docker container rm antispoof
        ;;
    test) 
        curl -F "image=@data/test_img.jpg" http://0.0.0.0:5000/detect-spoof
        docker logs antispoof
        curl -F "image=@data/test_img2.jpg" http://0.0.0.0:5000/detect-spoof
        docker logs antispoof
        ;;
esac