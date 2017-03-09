#! /usr/bin/python3
import sys
import cv2
import numpy as np
from pixelColorCounter import pixelColorCounter as pcc
from drawImgAndHist import drawImgAndHist as dih

class Main:

	path = ''
	img = ''
	count = None

	def __init__(self):
		if (len(sys.argv) > 2):
			print('Wrong number of parameters')
			sys.exit(1)

		if (len(sys.argv) == 2):
			self.path=sys.argv[1]
			self.load(True)

		while (True):
			self.switch(input("Choose:\n1) Load image\n2) Generete the image pixel's color counter(and relatives histogram)\n0) Exit program\n"))

	def load(self, loadFromCmd):
		if(loadFromCmd == False):
			self.path = input('Insert new path:\n')
		loadType = input("Choose image loading type:\n0) Grayscale Image\n1) Color image\n")
		self.img = cv2.imread(self.path, int(loadType))

	def histogram(self):
		self.count = pcc(self.img)
		showHist = input("Do you want to view the pixel's histogram of colors? [Y,n]\n") or 'y'
		if(showHist.lower() == 'y'):
			dih(self.img, self.count)

	def switch(self, argument):
		if(argument == '0'):
			sys.exit(0)
		fun, param = {
			'1': (self.load, False),
			'2': (self.histogram, None)
		}.get(argument)
		if(param is None):
			fun()
		else:
			fun(param)

if __name__ == '__main__':
	Main()
	
#cv.imshow('prova',img)
#cv.waitKey(1000)
