import cv2 
from itertools import product

def contrastBrightness(img):
	bottom = 0
	top = 255
	contrast = int(input("Insert the value of the contrast: "))
	brightness = int(input("Insert the value of the brightness: "))
	h,w,c=img.shape
	
	for channel,i,j in product(range(0,c), range(0,h), range(0,w))
		img[i,j,channel] = (img[i,j,channel]*contrast)+brightness
		if(img[i,j,channel]> top):
			img[i,j,channel] = 255
		elif(img[i,j,channel] < 0):
			img[i,j,channel] = 0
	return img
