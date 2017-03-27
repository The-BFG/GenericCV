import os, cv2, histogram
from neighborhoodOperators import * 
from linearPointOperators import *
from helpers import *
img = []

def features(mask):
	area = numpy.count_nonzero(mask)
	(masks, contours) = connected_components(mask)
	#print(len(masks),len(contours))
	return area, masks, contours

for f in os.listdir("data/train"):
		
	h = (cv2.imread("data/train/"+f, cv2.IMREAD_GRAYSCALE))
	if h is not None and h.ndim == 2:
		h = numpy.expand_dims(h, axis=2)
		img.append(h)
	
for f in os.listdir("data/test"):
	h = (cv2.imread("data/train/"+f, cv2.IMREAD_GRAYSCALE))
	if h is not None and h.ndim == 2:
		h = numpy.expand_dims(h, axis=2)
		img.append(h)
	
o = []
var = []

for i in range(len(img)):
	img[i] = threshold(img[i], otsu(histogram.build(img[i], 256)))
	
	
	_,_,var1 = features(img[i])
	var1= numpy.array(var1)
	print(var1.shape)
	var.append(var1[0])
	
cv2.namedWindow('Image',cv2.WINDOW_FREERATIO)
cv2.resizeWindow('Image', 100, 100)
cv2.imshow('Image', img[0])

cv2.waitKey()
cv2.destroyAllWindows()

cv2.namedWindow('Image',cv2.WINDOW_FREERATIO)
cv2.resizeWindow('Image', 100, 100)
cv2.imshow('Image', var[0])

cv2.waitKey()
cv2.destroyAllWindows()
