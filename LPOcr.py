from darknet_v3 import *
from logger import Logger

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

