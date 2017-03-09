import cv2 as cv
import numpy as np
from itertools import product

def ContrastStretching(img):
	m = np.min(img)
	M = np.max(img)
	#print(img.shape)
	h,w,c = img.shape
	for channel,i,j in product(range(0,c),range(0,h), range(0,w)):
		if(img[i,j,channel] >= m and img[i,j,channel] <= M):
			img[i,j,channel]= ((img[i,j,channel]-m)/(M-m)) *255
		elif(img[i,j,channel] < m):
			img[i,j,channel] = 0
		elif(img[i,j,channel] > M):
			img[i,j,channel] =255
	
	return img
