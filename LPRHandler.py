# 
# @author Felipe Serna
# @email damsog38@gmail.com
# @create date 2022-02-26 20:01:10
# @modify date 2022-03-03 14:19:39
# @desc Licence Plate Recognition system. this module integrates the LPR core and
# uses it to detect the licenses on an image (Using an License plate detector and an OCR), 
# then applies a simple centroid tracker and stores the detected labels of 
# the plate and determines given the history of labels which is the correct label of the plate.
# This class can be called on any algorithm to be used on different situations.

#*************************************************************************************************
#                                              Dependencies
#*************************************************************************************************

import argparse
import json
import pyfiglet
import iridi
from logger import Logger
from LPR import LPR
from functions import *
from simple_tracker.centroid_tracker import CentroidTracker

#*************************************************************************************************
#                                              Definitions
#*************************************************************************************************
class LPRHandler:
    def __init__(self, LOGGER_LEVEL="DEBUG", SHOW_TIME=True, detection_zone=None) -> None:
        self.LOGGER_LEVEL = LOGGER_LEVEL
        self.MODULE_NAME = "LPRHANDLER MODULE"
        if LOGGER_LEVEL=="DEBUG":
            self.logger = Logger("DEBUG", COLORED=True, TAG_MODULE=self.MODULE_NAME)
        else:
            self.logger = Logger("INFO", COLORED=True, TAG_MODULE=self.MODULE_NAME)

        # lpr_processor applies LPDetection and OCR on a single image to get the LPs on the image
        self.lpr_processor = LPR(detection_zone=detection_zone)

        # The CentroidTracker will assign an ID to each LP and will save the Labels
        # Remembering the list labels an LP has had and determining the correct with the
        # most frequent one.
        self.tracker = CentroidTracker()
    
    # Detects LPs on an image and gets their labels (OCR)
    def applyLPR(self, image, return_image=False):
        self.logger.debug("Applying LPR to image")
        if return_image:
            license_plates_json, img = self.lpr_processor.applyLPR(image, return_image=return_image)
            return license_plates_json, img
        else:
            license_plates_json = self.lpr_processor.applyLPR(image, return_image=return_image)
            return license_plates_json
    
    # Json to Lists handling data output
    def json_to_list(self, plates_json):
        bboxes = [ plate['bbox'] for plate in plates_json['plates'] ]
        labels = [ plate['label'] for plate in plates_json['plates'] ]
        return bboxes, labels
    
    # Lists to Json handling data output
    def list_to_json(self, boxes, labels):
        plates_json = {}
        plates_json["plates"] = [ {'bbox':box,'label':label,'id':id} for (id,box,label) in zip(boxes,boxes.values(), labels.values()) ]
        return plates_json

    # Updates the CentroidTracker using the new information gotten from the lpr_processor on a new image
    def update_tracker(self, plates_json, image=None):
        self.logger.debug("Updating Tracker and LP labels")
        boxes, labels = self.json_to_list(plates_json)
        _, boxes, labels = self.tracker.update( boxes,labels )
        if image is not None:
            image_drawn = self.draw_plates(boxes, labels, image)
            return self.list_to_json(boxes, labels), image_drawn
        else:
            return self.list_to_json(boxes, labels),_
    
    # Draws the information obtained and returns it for better visualization
    def draw_plates(self, boxes, labels, image):
        self.logger.debug("Drawing information to image")
        for (id,box, label) in zip(boxes,boxes.values(), labels.values()):
            cv2.rectangle(image, (box[0],box[1]),(box[2],box[3]), (128,128,0), 2)
            cv2.rectangle(image, (box[0],box[1]-20),(box[2], box[1] ), (128,128,0), -1)
            cv2.putText(image, f'Id:{id} L:{label}' , (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (54,51,53), 2)
        return image

def list2json(points_list):
    output_json = []
    for coordinate in points_list:
        point_json={}
        point_json["x"]=coordinate[0]
        point_json["y"]=coordinate[1]
        output_json.append(point_json)
    return output_json

#*************************************************************************************************
#                                              Main Function
#*************************************************************************************************

def main() -> None:
    # Displaying a cool title on the terminal
    moduleTitle = pyfiglet.figlet_format("L-P-R", font="isometric2", width=200)
    moduleSubtitle = pyfiglet.figlet_format("HANDLER", font="alligator2", width=300)

    iridi.print(moduleTitle, ["#8A2387", "#E94057", "#F27121"], bold=True)
    iridi.print(moduleSubtitle, ["#8A2387", "#E94057", "#F27121"], bold=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Media source. Video, Stream or Image")
    ap.add_argument("-iz", "--zone", required=False, help="""Points that determine the zone where to reduce the detection. as a json. Exmaple: 
                                                             Example: '[{"x": 10, "y": 10}, {"x": 100, "y": 120},{...}]' """)
    ap.add_argument("-dz", "--draw_zone", action="store_true", help="If selected allows to draw the input zone manually. this option ignores the flag -iz")
    ap.add_argument("-s", "--show_img", action="store_true", help="Shows image output")
    ap.add_argument("-d", "--debug", action="store_true", help="Sets Logger level to debug")
    args= vars(ap.parse_args())
    # Commad Example
    # python3 LPRHandler.py -i '/mnt/72086E48086E0C03/WORK_SPACE/Lpr/test_videos/test2.webm' -iz '[{"x": 150, "y": 217}, {"x": 97, "y": 308}, {"x": 561, "y": 299}, {"x": 551, "y": 227}, {"x": 150, "y": 217}]' -s
    # python3 LPRHandler.py -i '/mnt/72086E48086E0C03/WORK_SPACE/Lpr/test_videos/test2.webm' -dz -s


    MODULE_NAME = "LPR HANDLER MODULE"

    if args["debug"]:
        logger = Logger("DEBUG", COLORED=True,  TAG_MODULE=MODULE_NAME)
    else:
        logger = Logger("INFO", COLORED=True,  TAG_MODULE=MODULE_NAME)
    
    # Input Zone Handling
    if args["draw_zone"]:
        # If drawing the Zone manually
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
        # If the zone was input as an arg
        if args["zone"]:
            zone = json.loads(args["zone"])
            logger.info(f'Zone input: {zone}')

        else:
            logger.info("No zone input or drawn. applying detection on the whole image")
            zone = None
    
    # Creating our LPRHandler Object
    lpr_handler = LPRHandler(detection_zone=zone)

    #Openning Media and Iterating Video Frames
    media=cv2.VideoCapture(args["input"])
    while media.isOpened():
        ret, image = media.read()
        if not ret:
            logger.error('Error reading frame or video. exiting')
            break

        if args["show_img"]:
            # If the result images will be shown
            license_plates_json = lpr_handler.applyLPR(image)
            license_plates_json_updated, img = lpr_handler.update_tracker(license_plates_json, image)
            logger.info(f'LPRHandler Image result: {license_plates_json_updated}')
                
            cv2.namedWindow('Frame')
            cv2.imshow('Frame', img)

            fin = cv2.waitKey(1) & 0xFF
            if(fin == ord('q')):
                logger.info("Terminate key pressed. Closing program")
                break
        else:
            # If the result image wont be shown. information still output on the terminal
            license_plates_json = lpr_handler.applyLPR(image)
            license_plates_json_updated = lpr_handler.update_tracker(license_plates_json)
            logger.info(f'LPRHandler Image result: {license_plates_json_updated}')

if __name__ == '__main__':
    main()