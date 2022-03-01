import argparse
from logger import Logger
from LPR import LPR
from simple_tracker.centroid_tracker import CentroidTracker

class LPRHandler:
    def __init__(self) -> None:
        self.tracker = CentroidTracker()        
        self.lpr_processor = LPR()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Media source. Video, Stream or Image")
    ap.add_argument("-iz", "--zone", required=False, help="""Points that determine the zone where to reduce the detection. as a json. Exmaple: 
                                                             Example: '[{"x": 10, "y": 10}, {"x": 100, "y": 120},{...}]' """)
    ap.add_argument("-dz", "--draw_zone", action="store_true", help="If selected allows to draw the input zone manually. this option ignores the flag -iz")
    ap.add_argument("-s", "--show_img", action="store_true", help="Shows image output")
    ap.add_argument("-d", "--debug", action="store_true", help="Sets Logger level to debug")
    args= vars(ap.parse_args())
    # Commad Example
    # python3 LPR.py -i '/mnt/72086E48086E0C03/WORK_SPACE/Lpr/test_videos/test2.webm' -iz '[{"x": 150, "y": 217}, {"x": 97, "y": 308}, {"x": 561, "y": 299}, {"x": 551, "y": 227}, {"x": 150, "y": 217}]' -s
    # python3 LPR.py -i '/mnt/72086E48086E0C03/WORK_SPACE/Lpr/test_videos/test2.webm' -dz -s




if __name__ == '__main__':
    main()