from ctypes import *
import math
import random
import cv2
from src.keras_utils 			import load_model, detect_lp
from src.utils 					import im2single    
import numpy as  np
import time
import tensorflow as tf
from keras import backend as K
import keras
from functions import *
import shapely

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/mnt/72086E48086E0C03/nD/myProjects/Yolov3/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    wh = (im.w,im.h)
    free_image(im)
    free_detections(dets, num)
    return res,wh
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    SHOW_TIME = False
    RESIZE = 0.55
    net = load_net(b"data/ocr/ocr-net.cfg", b"data/ocr/ocr-net.weights", 0)
    meta = load_meta(b"data/ocr/ocr-net.data")
    lp_threshold = .5

    wpod_net_path = 'data/lp-detector/wpod-net_update1.h5'

    config = tf.ConfigProto( device_count = {'GPU': 0} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    with tf.device('/device:XLA_GPU:0'):
        #new_model = load_model('test_model.h5')
        wpod_net = load_model(wpod_net_path)
    #wpod_net = load_model(wpod_net_path)
    #print(k.tensorflow_backend._get_available_gpus())
    #print ('Searching for license plates using WPOD-NET')
    cap=cv2.VideoCapture('/mnt/72086E48086E0C03/WORK_SPACE/Lpr/LPR/lpr_test3.avi')

    ret, frame = cap.read()
    inputZones = []
    #inputZones = selectPolygonZone(frame,'green')
    #inputZones = inputZones[0]
    #polizone = Polygon(  [inputZones[0], inputZones[1], inputZones[2], inputZones[3]] )
    #zone_pts = np.array([ [inputZones[0][0],inputZones[0][1]] ,[inputZones[1][0],inputZones[1][1]] , [inputZones[2][0],inputZones[2][1]], [inputZones[3][0],inputZones[3][1]] ])
    while cap.isOpened():
        prev_time = time.time()

        
        ret, Ivehicle = cap.read()
        if not ret:
            print('Error reading frame or video finished')
            break
        height, width,_ = Ivehicle.shape
        im2show = Ivehicle.copy()
        ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)

        #print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
        prev_time_det = time.time()
        Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
        if SHOW_TIME:
            print('lp det time', time.time()-prev_time_det)
        #print(len(LlpImgs))
        ocr_ptime = time.time()
        #cv2.polylines(im2show, [zone_pts], True, (0,255,0), 2)
        for i in range(len(LlpImgs)):
            
            Ilp = LlpImgs[i]
            pts_pl = Llp[i].pts
            lp_points = int(pts_pl[0][0]*width),int(pts_pl[1][1]*height),int(pts_pl[0][2]*width),int(pts_pl[1][3]*height)
            lp_center = (int( (pts_pl[0][0]*width + pts_pl[0][2]*width)/2 ), int( ( pts_pl[1][1]*height+ pts_pl[1][3]*height)/2) )
            #if(polizone.contains( Point( lp_center)  ) ):
            #print(lp_points,lp_center)
            
            cv2.rectangle(im2show, (lp_points[0],lp_points[1]),(lp_points[2],lp_points[3]), (0,255,0), 2)


            #print(LlpImgs)
            
            Ilp=Ilp*255
            Ilp=Ilp.astype(np.uint8)
            #cv2.imshow("???",Ilp)
            #cv2.waitKey(10)
            
            cv2.imwrite('photo.jpg',Ilp)
            r = detect(net, meta, b"photo.jpg",0.4)
            
            #print(r)
            posicion = [0]*15
            letra= [0]*15
            cont=0
            for det in r[0]:
                #print(det[0]) #letra o numero placa
                #print(det[1])#Fiabilidad
                #print(det[2])#posición de la letra o numero

                posicion[cont]=det[2][0]
                letra[cont]=det[0].decode("utf-8")
                cont=cont+1
                box=[int(kk) for kk in det[2]]
                cv2.rectangle(
                    Ilp,
                    (int(box[0]-box[2]/2), int(box[1]-box[3]/2)),
                    (int(box[0]+box[2]/2), int(box[1]+box[3]/2)),
                    (0, 255, 0),
                    3
                )

                cv2.putText(Ilp,
                    str(det[0])[2], 
                    (int(box[0]), int(box[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    1
                )
            cv2.imshow("BITCH",Ilp)
            cv2.waitKey(0)
            #print(posicion)
            while 0 in posicion:
                posicion.remove(0)
            desorganizado=posicion.copy()
            posicion.sort() #posicion organizada
            tam=len(posicion)
            #print("posicion organizada")
            #print(posicion)
            #print("Letra")
            #print(letra)
            matricula= [0]*15
            cont1=0
            while cont1 < tam:
                #print(desorganizado.index(posicion[cont1]))
                matricula[cont1]=letra[desorganizado.index(posicion[cont1])]
                cont1=cont1+1
            print(matricula)
        if SHOW_TIME:
            print('yolo time:', time.time() - ocr_ptime)
        im2show = cv2.resize(im2show, ( int(im2show.shape[1]*RESIZE),int(im2show.shape[0]*RESIZE) ), interpolation = cv2.INTER_AREA) 
        cv2.imshow("origi",im2show)
        cv2.waitKey(2)
        if SHOW_TIME:
            print("FPS: ", 1.0 / (time.time() - prev_time))
        
        #for det in r[0]:
