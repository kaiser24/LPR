import argparse
from ctypes import *
import json
from tkinter.messagebox import NO
import cv2
import numpy as  np
import time
from functions import *
from LPOcr import LPOcr
from LPDetector import LPDetector
from logger import Logger

class LPR:
    def __init__(self, LOGGER_LEVEL="DEBUG", SHOW_TIME=True, detection_zone=None) -> None:
        self.LOGGER_LEVEL = LOGGER_LEVEL
        self.MODULE_NAME = "LPR MODULE"

        self.SHOW_TIME = SHOW_TIME

        if detection_zone is not None:
            self.setDetectionZone(detection_zone)

        self.lpocr_processor = LPOcr()
        self.lp_detector = LPDetector()

        if LOGGER_LEVEL=="DEBUG":
            self.logger = Logger("DEBUG", COLORED=True, TAG_MODULE=self.MODULE_NAME)
        else:
            self.logger = Logger("INFO", COLORED=True, TAG_MODULE=self.MODULE_NAME)

        self.json_output = {}

    def setDetectionZone(self, zone):
        self.detectionZone = zone
        self.detectionPolygon = self.detectionZone2Polyzone(zone)
        self.detectionZonePoints = self.detectionZone2Points(zone)
    
    def getDetectionZone(self):
        return self.detectionZone
    
    def detectionZone2Polyzone(self, zone):
        inputZones = [ (point["x"],point["y"]) for point in zone ]
        polyzone = Polygon(  [inputZones[0], inputZones[1], inputZones[2], inputZones[3]] )
        return polyzone

    def detectionZone2Points(self, zone):
        inputZones = [ (point["x"],point["y"]) for point in zone ]
        zonePoints = np.array([ [inputZones[0][0],inputZones[0][1]] ,[inputZones[1][0],inputZones[1][1]] , [inputZones[2][0],inputZones[2][1]], [inputZones[3][0],inputZones[3][1]] ])
        return zonePoints

    def applyLPR(self, image):

        # Initiallizations
        self.json_output = {}
        output_array = []
        
        height, width,_ = image.shape

        # Applying LP Detector
        prev_time_det = time.time()
        self.logger.debug("Applying LP detection on image")
        license_pnts_list,license_img_list,_ = self.lp_detector.applyLPDetection(image, 0.5)
        if self.SHOW_TIME:
            self.logger.debug(f'LPDetection time {time.time()-prev_time_det}')
        

        for i,license_plate_img in enumerate(license_img_list):
            self.logger.debug(f'Licence plate: {i}')
            license_plate_img = license_img_list[i]
            license_pnts = license_pnts_list[i].pts
            lp_points = int(license_pnts[0][0]*width),int(license_pnts[1][1]*height),int(license_pnts[0][2]*width),int(license_pnts[1][3]*height)

            self.logger.debug(f'Licence plate: {i} points {lp_points}')

            #cv2.rectangle(im2show, (lp_points[0],lp_points[1]),(lp_points[2],lp_points[3]), (0,255,0), 2)
            
            license_plate_img=license_plate_img*255
            license_plate_img=license_plate_img.astype(np.uint8)
            
            cv2.imwrite('photo.jpg',license_plate_img)
            prev_time_ocr = time.time()
            ocrResult = self.lpocr_processor.applyOCR(b"photo.jpg",0.4)
            if self.SHOW_TIME:
                self.logger.debug(f'LPOCR time {time.time() - prev_time_ocr}')
            
            prev_time_lpro = time.time()


            if self.SHOW_TIME:
                self.logger.debug(f'LP Reorganized time {time.time() - prev_time_lpro}')
            
            lplate = { "label" : " ".join(map(str,ocrResult)), "bbox" : (lp_points) }
            self.logger.debug(f'LP image result {lplate}')

            output_array.append(lplate)     
        
        self.json_output = {"plates" : output_array }

        return self.json_output

def list2json(points_list):
    output_json = []
    for coordinate in points_list:
        point_json={}
        point_json["x"]=coordinate[0]
        point_json["y"]=coordinate[1]
        output_json.append(point_json)
    return output_json
    

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Media source. Video, Stream or Image")
    ap.add_argument("-iz", "--zone", required=False, help="""Points that determine the zone where to reduce the detection. as a json. Exmaple: 
                                                             Example: '[{"x": 10, "y": 10}, {"x": 100, "y": 120},{...}]' """)
    ap.add_argument("-dz", "--draw_zone", action="store_true", help="If selected allows to draw the input zone manually. this option ignores the flag -iz")
    ap.add_argument("-d", "--debug", action="store_true", help="Sets Logger level to debug")
    args= vars(ap.parse_args())
    # Commad Example
    # python3 LPR.py -i '/mnt/72086E48086E0C03/WORK_SPACE/Lpr/test_videos/test2.webm' -iz '[{"x": 150, "y": 217}, {"x": 97, "y": 308}, {"x": 561, "y": 299}, {"x": 551, "y": 227}, {"x": 150, "y": 217}]'
    # python3 LPR.py -i '/mnt/72086E48086E0C03/WORK_SPACE/Lpr/test_videos/test2.webm' -dz

    MODULE_NAME = "LPR STANDALONE MODULE"

    if args["debug"]:
        logger = Logger("DEBUG", COLORED=True,  TAG_MODULE=MODULE_NAME)
    else:
        logger = Logger("INFO", COLORED=True,  TAG_MODULE=MODULE_NAME)

    # Input Zone Handling
    if args["draw_zone"]:
        logger.info(f'Draw Zone selected. ignoring Zone input')

        media=cv2.VideoCapture(args["input"])
        ret, frame = media.read()

        inputZones = selectPolygonZone(frame,'green')
        inputZones = inputZones[0]

        zone=list2json(inputZones)
        media.release()

        logger.debug(f'As list zone {inputZones}')
        logger.info(f'Drawn Zone: {zone}')
    else:
        if args["zone"]:
            zone = json.loads(args["zone"])
            logger.info(f'Zone input: {zone}')

        else:
            logger.info("No zone input or drawn. applying detection on the whole image")
            zone = None


    SHOW_TIME = True
    RESIZE = 0.55

    lpr = LPR(detection_zone=zone)
    media=cv2.VideoCapture(args["input"])

    while media.isOpened():
        ret, image = media.read()
        if not ret:
            logger.error('Error reading frame or video. exiting')
            break
        im2show = image.copy()

        #cv2.polylines(im2show, [zone_pts], True, (0,255,0), 2)

        license_plates_json = lpr.applyLPR(image)
        logger.debug(f'LPR Image result: {license_plates_json}')
            
        im2show = cv2.resize(im2show, ( int(im2show.shape[1]*RESIZE),int(im2show.shape[0]*RESIZE) ), interpolation = cv2.INTER_AREA) 
        cv2.imshow("origi",im2show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
