import cv2, numpy as np
from roipoly.roipoly import RoiPoly
from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledlength = int(length * iteration // total)
    progessbar = fill * filledlength + '-' * (length - filledlength)
    print('\r%s |%s| %s%% %s' % (prefix, progessbar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def set_color(carType):
    if carType == "car":
        return (235, 52, 128)
    if carType == "bus":
        return (191, 181, 99)
    if carType == "truck":
        return (99, 191, 174)
    if carType == "motorbike":
        return (52, 137, 235)

def findPoint(rect, point):
    
    if (point[0] > rect[0] and point[0] < rect[2] and

            point[1] > rect[1] and point[1] < rect[3]):
        return True
    else:
        return False

def selectPolygonZone(image,color):
    zones = []
    
    while True:
        try:
            plt.figure(figsize=(12,8))
            
            plt.imshow(image, aspect='auto',cmap = 'gray')
            zone = RoiPoly(color = color)
            zone.x[0] = zone.x[-1]
            zone.y[0] = zone.y[-1]
            zone.x = list(map(int, zone.x))
            zone.y = list(map(int, zone.y))
            polyZone = []
                        
            for point in zip(zone.x,zone.y):
                polyZone.append((point))

            pts = np.array(polyZone, np.int32)
            pts = pts.reshape((-1,1,2))

            if color == 'red':
                cv2.polylines(image,[pts],True,(255,0,0),2)
            else:
                cv2.polylines(image,[pts],True,(0,255,0),2)

        except:
            break
        zones.append(polyZone)

    return zones


def containPoint(zone,point):
    p = Point(point[0],point[1])
    polygon = Polygon(zone)
    
    return polygon.contains(p) 

if __name__ == "__main__":
    image = cv2.imread("cam.jpg")
    point = (768, 423)

    for poly in selectPolygonZone(image,'red'):        
        print(containPoint(poly,point))
    
    #for entrada in selectInputZones(image):
    #    print(findPoint(entrada, (320, 323)))
    #selectPolygonZoneInput(image)