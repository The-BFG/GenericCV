#! /usr/bin/python3
import sys
import cv2
import numpy
from pixelColorCounter import pixelColorCounter as pcc
from drawImgAndHist import drawImgAndHist as dih
from linearPointOperator import contrastBrightness
from ContrastStretching import ContrastStretching as cs

class Main:

	path = ''
	img = None
	count = None

	def __init__(self):
		if (len(sys.argv) > 2):
			print('Wrong number of parameters')
			sys.exit(1)

		if (len(sys.argv) == 2):
			self.path=sys.argv[1]
			self.load(True)

		while (True):
			self.switch(input("Choose:\n1) Load image\n2) Show image\n3) Generete the image pixel's color counter (and relatives histogram)\n4) Change contrast and brightness\n5) Contrast stretching\n0) Exit program\n"))

	def load(self, loadFromCmd):
		if(loadFromCmd == False):
			self.path = input('Insert new path: ')
		loadType = input("Choose image loading type:\n0) Grayscale Image\n1) Color image\n")
		self.img = cv2.imread(self.path, int(loadType))
		if(self.img.ndim == 2):
			self.img = numpy.expand_dims(self.img, axis=2)
	
	def show(self):
		cv2.imshow('Image',self.img)
		cv2.waitKey()

	def histogram(self):
		self.count = pcc(self.img)
		showHist = input("Do you want to view the pixel's histogram of colors? [Y,n] ") or 'y'
		if(showHist.lower() == 'y'):
			dih(self.img, self.count)

	def changeContrastBrightness(self):
		self.img = contrastBrightness(self.img)
		
	def stretchContrast(self):
		self.img = cs(self.img)

	def switch(self, argument):
		if(argument == '0'):
			sys.exit(0)
		fun, param = {
			'1': (self.load, False),
			'2': (self.show, None),
			'3': (self.histogram, None),
			'4': (self.changeContrastBrightness, None),
			'5': (self.stretchContrast, None)
		}.get(argument)
		if(param is None):
			fun()
		else:
			fun(param)

if __name__ == '__main__':
	Main()
	
#cv.imshow('prova',img)
#cv.waitKey(1000)
