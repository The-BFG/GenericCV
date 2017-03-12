from itertools import product

def threshold(img, t):
	if(len(img.shape)==2):
		h,w=img.shape
		c=1
	elif(len(img.shape)==3):
		h,w,c=img.shape
	for channel,i,j in product(range(0,c),range(0,h),range(0,w)):
		if(len(img.shape)==2):
			img[i,j] =  255 if(img[i,j] > t) else 0
		elif(len(img.shape)==3):
			img[i,j,channel] =  255 if(img[i,j,channel] > t) else 0
	return img
