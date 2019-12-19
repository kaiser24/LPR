import sys, os
import keras
import cv2
import traceback

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes



if __name__ == '__main__':

	lp_threshold = .5

	wpod_net_path = 'data/lp-detector/wpod-net_update1.h5'
	wpod_net = load_model(wpod_net_path)

	print ('Searching for license plates using WPOD-NET')
	cap=cv2.VideoCapture('/home/santi/Downloads/LPR10.mp4')
	while True:

		
		ret, Ivehicle = cap.read()

		ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)),608)
		print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

		Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

		if len(LlpImgs):
			Ilp = LlpImgs[0]
			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
			Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
			cv2.imshow("???",Ilp)
			cv2.waitKey(10)

		cv2.imshow("origi",Ivehicle)
		cv2.waitKey(10)



