from ctypes import *
import cv2
import numpy as  np
import time
from functions import *
from LPOcr import LPOcr
from LPDetector import LPDetector
from logger import Logger

class LPR:
    def __init__(self, LOGGER_LEVEL="DEBUG", SHOW_TIME=True) -> None:
        self.LOGGER_LEVEL = LOGGER_LEVEL
        self.MODULE_NAME = "LPR MODULE"

        self.lpocr_processor = LPOcr()
        self.lp_detector = LPDetector()

        if LOGGER_LEVEL=="DEBUG":
            self.logger = Logger("DEBUG", COLORED=True, TAG_MODULE=self.MODULE_NAME)
        else:
            self.logger = Logger("INFO", COLORED=True, TAG_MODULE=self.MODULE_NAME)

    def applyLPR(self, image):
        height, width,_ = image.shape

        # Applying LP Detector
        prev_time_det = time.time()
        self.logger.debug("Applying LP detection on image")
        license_pnts_list,license_img_list,_ = lpr.lp_detector.applyLPDetection(image, 0.5)
        if SHOW_TIME:
            lpr.logger.debug(f'LPDetection time {time.time()-prev_time_det}')
        

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
            ocrResult, ocrResult2 = self.lpocr_processor.applyOCR(b"photo.jpg",0.4)
            if SHOW_TIME:
                lpr.logger.debug(f'LPOCR time {time.time() - prev_time_ocr}')
            
            prev_time_lpro = time.time()


            if SHOW_TIME:
                lpr.logger.debug(f'LP Reorganized {time.time() - prev_time_lpro}')
            
            self.logger.debug(f'LP reorganized {ocrResult, ocrResult2}')
            

if __name__ == "__main__":

    SHOW_TIME = True
    RESIZE = 0.55

    lpr = LPR()

    cap=cv2.VideoCapture('/mnt/72086E48086E0C03/WORK_SPACE/Lpr/test_videos/test2.webm')

    ret, frame = cap.read()
    inputZones = []
    inputZones = selectPolygonZone(frame,'green')
    inputZones = inputZones[0]
    polizone = Polygon(  [inputZones[0], inputZones[1], inputZones[2], inputZones[3]] )
    zone_pts = np.array([ [inputZones[0][0],inputZones[0][1]] ,[inputZones[1][0],inputZones[1][1]] , [inputZones[2][0],inputZones[2][1]], [inputZones[3][0],inputZones[3][1]] ])
    
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            lpr.logger.error('Error reading frame or video. exiting')
            break
        im2show = image.copy()

        cv2.polylines(im2show, [zone_pts], True, (0,255,0), 2)

        lpr.applyLPR(image)
            
        im2show = cv2.resize(im2show, ( int(im2show.shape[1]*RESIZE),int(im2show.shape[0]*RESIZE) ), interpolation = cv2.INTER_AREA) 
        cv2.imshow("origi",im2show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        #cv2.waitKey(0)
        
        #for det in r[0]:
