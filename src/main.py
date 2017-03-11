#!/usr/bin/python3
import sys, getopt, cv2, numpy
import histogram

from linearPointOperators import *
from neighborhoodOperators import *

class Main:

	def __init__(self, argv):

		self.path, self.loadType, self.histogram = None, None, None

		self.options = {"-i": "self.path", "-t": "self.loadType"}

		self.menu = {
			'0': ("Exit program", sys.exit),
			'1': ("Load image", self.load),
			'2': ("Show image", self.show),
			'3': ("Generate the image histogram", self.createHistogram),
			'4': ("Change contrast and brightness", self.changeContrastBrightness),
			'5': ("Apply contrast stretching", self.stretchContrast),
			'6': ("Make a threshold", self.makeThreshold),
			'7': ("Otzu threashold on the image", self.makeOtsu),
			'8': ("Convolution", self.performConvolution)

		}

		try:
			opts, args = getopt.getopt(argv, "i:t:")
		except getopt.GetoptError:
			print('main.py -i <inputfile> -t <loadtype>')
			sys.exit(2)
		for opt, arg in opts:
			exec(self.options[opt] + ' = arg')

		if self.path is not None:
			self.load(self.path, self.loadType)

		while True:
			print("Choose:\n" + "\n".join("%d) %s" % (int(k), v[0]) for k, v in sorted(self.menu.items(),key=lambda a: int(a[0]) if(int(a[0])>0) else sys.maxsize)))
			self.menu[input()][1]()

	def load(self, path = None, loadType = None):
		self.path = path if path is not None else input('Insert new path: ')
		self.loadType = loadType if loadType is not None else input("Choose image loading type:\n0) Grayscale Image\n1) Color image\n")
		self.img = cv2.imread(self.path, int(self.loadType))
		print(self.img.dtype)
		if self.img.ndim == 2:
			self.img = numpy.expand_dims(self.img, axis=2)

	def show(self):
		cv2.imshow('Image', self.img)
		cv2.waitKey()
		cv2.destroyAllWindows()

	def createHistogram(self):
		self.histogram = histogram.build(self.img)
		if (input("Do you want to visualize the histogram? [Y,n] ").lower() or 'y') == 'y':
			histogram.display(self.img, self.histogram)

	def changeContrastBrightness(self):
		self.img = contrastBrightness(self.img)
		print(self.img.shape)

	def stretchContrast(self):
		self.img = contrastStreching(self.img)

	def makeThreshold(self):
		self.img = threshold(self.img, int(input("Insert the threshold: ")))

	def makeOtsu(self):
		self.histogram = histogram.build(self.img, bins=256)
		self.img = threshold(self.img, otsu(self.histogram))

	def performConvolution(self):
		self.img = convolution(self.img)

if __name__ == "__main__":
	Main(sys.argv[1:])