# 
# @author Felipe Serna
# @email damsog38@gmail.com
# @create date 2022-02-26 20:01:10
# @modify date 2022-03-03 14:19:39
# @desc Licence Plate Recognition system. this module applies OCR to an image. The algorithm
# consists of an Object Detection algorithm (YOLO) trained to detect 26 objects which are the letters
# of the latin alphabet and this way detecting the characters on the image. it was specially trained
# on license plate text. the characters are then reorganized and sent as output.

#*************************************************************************************************
#                                              Dependencies
#*************************************************************************************************

from darknet_v3 import *
from logger import Logger

#*************************************************************************************************
#                                              Definitions
#*************************************************************************************************

class LPOcr:
    def __init__(self, model="darknetv3", LOGGER_LEVEL="DEBUG") -> None:
        self.MODULE_NAME = "LICENCE PLATE OCR"
        if LOGGER_LEVEL=="DEBUG":
            self.logger = Logger("DEBUG", COLORED=True, TAG_MODULE= self.MODULE_NAME)
        else:
            self.logger = Logger("INFO", COLORED=True, TAG_MODULE= self.MODULE_NAME)

        self.logger.info("Creating LP OCR processor")
        if model=="darknetv3":
            self.logger.info("OCR backend model darknet v3 selected")
            self.net = load_net(b"data/ocr/ocr-net.cfg", b"data/ocr/ocr-net.weights", 0)
            self.meta = load_meta(b"data/ocr/ocr-net.data")
    
    def reorganizeCharacters(self, character_list):
        position = []
        letter = []
        for det in character_list[0]:
            position.append(det[2][0])
            letter.append(det[0].decode("utf-8"))
        
        desorganized=position.copy()
        position.sort()

        licencePlate = desorganized.copy()

        for i in range(len(position)):
            licencePlate[i]=letter[desorganized.index(position[i])]

        return licencePlate

    def applyOCR(self, image_src, threshold):
        self.logger.debug(f'Image route {image_src}')
        ocrResult = detect(self.net, self.meta, image_src, threshold)
        #self.logger.debug(f'OCR result: {result}')

        result = self.reorganizeCharacters(ocrResult)
        return result

