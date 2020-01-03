#!/usr/bin/env bash

case "$1" in
    build)
        docker build --rm -t face_detection:latest .
        docker run -d -p 5000:8000 --name=face_detect face_detection
        ;;
    kill)
        docker stop face_detect
        docker container rm face_detect
        ;;
    test) 
        curl -F "image=@data/test_img.jpg" http://0.0.0.0:5000/detect-face
        docker logs face_detect
        curl -F "image=@data/test_img2.jpg" http://0.0.0.0:5000/detect-face
        docker logs face_detect
        ;;
esac