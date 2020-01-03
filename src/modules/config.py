import os
import logging
from glob import glob

# Logger --------------------------------------------------------------------------------------
try:
    import colorlog
    HAVE_COLORLOG = True
except ImportError:
    HAVE_COLORLOG = False


def create_logger():
    """
        Setup the logging environment
    """
    log = logging.getLogger()  # root logger
    log.setLevel(logging.INFO)
    format_str = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    if HAVE_COLORLOG and os.isatty(2):
        cformat = '%(log_color)s' + format_str
        colors = {'DEBUG': 'reset',
                  'INFO': 'reset',
                  'WARNING': 'bold_yellow',
                  'ERROR': 'bold_red',
                  'CRITICAL': 'bold_red'}
        formatter = colorlog.ColoredFormatter(cformat, date_format,
                                              log_colors=colors)    
    else:
        formatter = logging.Formatter(format_str, date_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return logging.getLogger(__name__)


logger = create_logger()

# LOCAL WORKING DIRECTORIES --------------------------------------------------------------------------------
WORKING_DIR = '/usr/local/spoofing_detection_api/src'

## MODELS DIRECTORIES
MODELS_DIR = os.path.join(WORKING_DIR, 'models')

NN_MODELS_DIR = os.path.join(MODELS_DIR, 'nn_models')

NN_WEIGHTS_DIR = os.path.join(MODELS_DIR, 'nn_pretrained_weights')

LABELS_DIR = os.path.join(MODELS_DIR, 'labels')

DETECTORS_DIR = os.path.join(MODELS_DIR, 'detectors')

def find_model(folder, extension):
    """
    This is assuming there is only one of each extension each folder, which should be kept this case all the time.
    """
    extension = '*.' + extension
    dir = glob(os.path.join(folder, extension))
    return dir[0]