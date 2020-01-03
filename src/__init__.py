from flask import Flask, request, jsonify, json, make_response
from werkzeug.utils import secure_filename

from . import predict

import os

import numpy as np

from .modules import config as cfg

app = Flask(__name__)


UPLOAD_FOLDER = 'upload_imgs'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

logger = cfg.create_logger()


@app.route('/')
def ping():
    return jsonify(True)


@app.route('/detect-spoof', methods=['POST'])
def detect_face():
    if request.method == 'POST':
        if 'image' not in request.files:
            # to be implemented
            return False

        file = request.files['image']

        if file.filename == '':
            # to be implemented
            return False

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            os.mkdir(UPLOAD_FOLDER) if not os.path.exists(
                UPLOAD_FOLDER) else None

            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            result = predict.classify_image(save_path)

            # clean up
            os.remove(save_path)
            os.rmdir(UPLOAD_FOLDER)

            if result:
                # return result
                result['valid'] = True
                return json.dumps(result, default=convert)
            else:
                return jsonify({'spoof': False,
                                'runtime': None,
                                'valid': False
                                })

# Sub Functions # --------------------------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError
