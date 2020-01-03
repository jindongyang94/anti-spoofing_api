# spoofing_detection_api

The API is used to be deployed as a microservice on cloud (e.g. AWS) where an image is given and the result will returned as such:  

{  
    valid : < whether the result is applicable for the image or not. If it is not, the result should be disregarded. >,  
    spoof: < whether the image is spoofed (True) or not (False) >,  
    runtime: < the duration taken to reach the conclusion >  
}

The Makefile has all the needed functions to run and test in docker locally.  

Folder Structure:

```bash
.  
├── Dockerfile  
├── Makefile  
├── README.md  
├── config  
│   └── gunicorn.conf.py  
├── data  
├── entrypoint.sh  
├── misc  
├── requirements.txt  
├── scripts  
│   └── run.sh  
└── src  
    ├── __init__.py  
    ├── models  
    │   ├── detectors  
    │   │   ├── face_RFB  
    │   │   │   ├── RFB-320.caffemodel  
    │   │   │   └── RFB-320.prototxt  
    │   │   ├── face_detector  
    │   │   │   ├── deploy.prototxt  
    │   │   │   └── res10_300x300_ssd_iter_140000.caffemodel  
    │   │   └── haarcascade_eye.xml  
    │   ├── labels  
    │   │   └── le.pickle  
    │   └── nn_models  
    │       └── vgg16_pretrained.model  
    ├── modules  
    │   ├── __init__.py  
    │   ├── config.py  
    │   └── nn_predict_helper.py  
    └── predict.py  
```
