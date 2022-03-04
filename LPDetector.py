# 
# @author Felipe Serna
# @email damsog38@gmail.com
# @create date 2022-02-26 20:01:10
# @modify date 2022-03-03 14:19:39
# @desc Licence Plate Recognition system. this module applies an Object detection algorithm
# trained only on detecting license plates which then can be send to another module to 
# do other analysis, like applying OCR.

#*************************************************************************************************
#                                              Dependencies
#*************************************************************************************************

from logger import Logger
from src.keras_utils import load_model
from src.keras_utils import detect_lp
from src.utils import im2single    
import tensorflow as tf
import keras

#*************************************************************************************************
#                                              Definitions
#*************************************************************************************************

class LPDetector:
    def __init__(self, model="WPOD-NET", device="GPU", LOGGER_LEVEL="DEBUG") -> None:
        self.MODULE_NAME = "LICENCE PLATE OCR"
        if LOGGER_LEVEL=="DEBUG":
            self.logger = Logger("DEBUG", COLORED=True, TAG_MODULE= self.MODULE_NAME)
        else:
            self.logger = Logger("INFO", COLORED=True, TAG_MODULE= self.MODULE_NAME)

        self.logger.info("Creating LP Detector")
        
        self.device = device
        
        self.wpod_net_path = 'data/lp-detector/wpod-net_update1.h5'
        self.config = tf.ConfigProto( device_count = {'GPU': 0} ) 
        self.sess = tf.Session(config=self.config) 
        keras.backend.set_session(self.sess)

        if self.device == "GPU":
            self.logger.info("LPDetector loading on GPU")
            #self.logger.info(f'Available devices {keras.tensorflow_backend._get_available_gpus()}')
            with tf.device('/device:XLA_GPU:0'):
                self.wpod_net = load_model(self.wpod_net_path)
        else:
            self.logger.info("LPDetector loading on CPU")
            self.wpod_net = load_model(self.wpod_net_path)
    
    def applyLPDetection(self, image, threshold):
        ratio = float(max(image.shape[:2]))/min(image.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)

        result = detect_lp(self.wpod_net,im2single(image),bound_dim,2**4,(240,80),threshold)
        return result