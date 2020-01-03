import os
import pickle
import time

import cv2
from keras.models import load_model

from .modules import nn_predict_helper as nn
from .modules import config as cfg


def classify_image(img_path):
    """
    From a image folder location:
    1. Create a real and fake image folder in the current image folder itself. (Only if there aren't such a folder)
    2. Classify the images into real and fake and store them within the created folders. 
    """

    args = {
        'detector': 'face_RFB',
        'model': 'vgg16_pretrained.model',
        'le': 'le.pickle',
        'confidence': 0.9
    }

    result = {
        'runtime': 0
    }

    start_time = time.time()

    # Load Models
    # Load our serialized face detector from disk
    print("[INFO] loading face detector...")
    face_detector_path = os.path.join(cfg.DETECTORS_DIR, args['detector'])
    protoPath = cfg.find_model(face_detector_path, 'prototxt')
    modelPath = cfg.find_model(face_detector_path, "caffemodel")
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Load the liveness detector model and label encoder from disk
    print("[INFO] loading liveness detector...")
    classifiermodelpath = os.path.join(cfg.NN_MODELS_DIR, args['model'])
    model = load_model(classifiermodelpath)
    le = pickle.loads(
        open(os.path.join(cfg.LABELS_DIR, args["le"]), "rb").read())

    frame = cv2.imread(img_path)
    if args['detector'] == 'face_RFB':
        frame, finally_fake, detected_faces = nn.label_with_face_detector_ultra(
            frame, net, model, le, args['confidence'])
    else:
        frame, finally_fake, detected_faces = nn.label_with_face_detector_original(
            frame, net, model, le, args['confidence'])

    runtime = time.time() - start_time

    # Return result based on whether it is fake, real or noface
    if detected_faces == 0:
        return False

    elif finally_fake:
        result['spoof'] = True
    else:
        result['spoof'] = False

    result['runtime'] = runtime

    return result
