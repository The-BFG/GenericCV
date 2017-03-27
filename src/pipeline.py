import os, histogram, cv2
from linearPointOperators import *
from neighborhoodOperators import *

def features(mask):
	area = numpy.count_nonzero(mask)


img = []

for f in os.listdir("data/train"):
	print(f)
	h = cv2.imread("data/train/"+f, cv2.IMREAD_GRAYSCALE)
	if h is not None and h.ndim == 2:
		h = numpy.expand_dims(h, axis=2)
		img.append(h)


for f in os.listdir("data/test"):
	h = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
	if h.ndim == 2:
		h = numpy.expand_dims(h, axis=2)
	img.append(h)

for i in range(len(img)):
	img[i] = threshold(img[i],otsu(histogram.build(img[i],256)))
	cv2.namedWindow('Image', cv2.WINDOW_FREERATIO)
	cv2.resizeWindow('Image', 100, 100)
	cv2.imshow('Image', img[i])

	cv2.waitKey()
	cv2.destroyAllWindows()
