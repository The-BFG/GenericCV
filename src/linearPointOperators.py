import cv2, numpy
from itertools import product


def contrastBrightness(img, contrast = None, brightness = None):

	contrast = contrast if contrast is not None else float(input("Insert the value of the contrast: "))
	brightness = brightness if brightness is not None else float(input("Insert the value of the brightness: "))

	return (img * contrast + brightness).astype(numpy.uint8)

def contrastStreching(img, m = None, M = None, nm = None, nM = None):
	m = m if m is not None else int(input("Insert the minimum value: [min] ") or numpy.min(img))
	M = M if M is not None else int(input("Insert the maximum value: [max] ") or numpy.max(img))
	nm = nm if nm is not None else int(input("Insert the minimum value: [0] ") or 0)
	nM = nM if nM is not None else int(input("Insert the maximum value: [255] ") or 255)

	r = (nM - nm) / (M - m)

	return contrastBrightness(img, r, nm - m * r)


def threshold(img, threshold):
	return ((img > threshold) * 255).astype(numpy.uint8)

def otsu(histogram):
	sigma = []
	for t in range(len(histogram)):
		q_L = sum(histogram[:t])
		q_H = sum(histogram[t:])
		if(q_H > 0 and q_L > 0):
			miu_L = sum(numpy.multiply(histogram[:t], range(t))) / q_L;
			miu_H = sum(numpy.multiply(histogram[t:], range(t,len(histogram)))) / q_H;
			sigma.append(q_L * q_H * ((miu_L - miu_H)**2))
	return sigma.index(max(sigma))
