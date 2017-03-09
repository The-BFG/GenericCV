import cv2 as cv
import numpy as np
from itertools import product

def pixelColorCounter(img):
	bins = input('Insert number of bins: ')
	return pixelColorCounterWithBins(img,int(bins))

def pixelColorCounterWithBins(img, bins):
	tot=256
	if(len(img.shape)==2):
		h,w=img.shape
		c=1
	elif(len(img.shape)==3):
		h,w,c=img.shape

	#print(type(bins), type(c))
	count=np.zeros(int(bins*c))
	for channel,i,j in product(range(0,c),range(0,h),range(0,w)):
		if(len(img.shape)==2):
			index=int(img[i][j]/(tot/bins))
		if(len(img.shape)==3):
			index=int(img[i][j][channel]/(tot/bins))
		count[index+channel*bins]+=1
	count/=(int(h*w))
	return count




