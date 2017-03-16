#!/usr/bin/python3
import sys, getopt, cv2, numpy
import histogram

from linearPointOperators import *
from neighborhoodOperators import *

class Main:

	def __init__(self, argv):

		self.path, self.loadType, self.histogram = None, None, None

		# TODO: Aggiungere opzione "-c commandi:..."
		self.options = {"-i": "self.path", "-t": "self.loadType"}

		# TODO: Aggiungere sottomenu
		self.menu = {
			'0': ("Exit program", sys.exit),
			'1': ("Load image", self.load),
			'2': ("Show image", self.show),
			'3': ("Generate the image histogram", self.createHistogram),
			'4': ("Change contrast and brightness", self.changeContrastBrightness),
			'5': ("Apply contrast stretching", self.stretchContrast),
			'6': ("Make a threshold", self.makeThreshold),
			'7': ("Otzu threashold on the image", self.makeOtsu),
			'8': ("Convolution", self.performConvolution),
			'9': ("Sobel", self.sobel),
			'10': ("Canny", self.canny)
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
		if self.img.ndim == 2:
			self.img = numpy.expand_dims(self.img, axis=2)

	def show(self):

		cv2.namedWindow('Image',cv2.WINDOW_FREERATIO)
		cv2.resizeWindow('Image', 100, 100)
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

	# TODO: Add a way to choose kernel
	def performConvolution(self):
		self.img = convolution(self.img, kernel("gauss",8))

	def sobel(self):
		gradx = convolution(self.img, kernel("sobelx"))
		grady = convolution(self.img, kernel("sobely"))
		self.img = numpy.sqrt(gradx ** 2 + grady ** 2).astype(numpy.uint8)
		self.img = contrastStreching(self.img,numpy.min(self.img), numpy.max(self.img), 0, 255)

	# TODO: Not working
	def canny(self):

		gradx = convolution(self.img, kernel("sobelx"))
		grady = convolution(self.img, kernel("sobely"))
		grad = numpy.sqrt(gradx ** 2 + grady ** 2).astype(numpy.uint8)

		tan = numpy.arctan(grady / gradx)
		h, w, c = numpy.array(grad).shape

		max = numpy.max(grad)
		tl = 0.90 * max
		th = 0.99 * max

		for i in range(1,h-1):
			for j in range(1,w-1):
				m = tan[i,j]
				if gradx[i,j] > grady[i,j]:
					if m > 0:
						gradP1 = grad[i + 1, j - 1] * m + grad[i + 1, j] * (1 - m)
						gradP2 = grad[i - 1, j + 1] * m + grad[i - 1, j] * (1 - m)
					else:
						gradP1 = grad[i + 1, j + 1] * m + grad[i + 1, j] * (1 - m)
						gradP2 = grad[i - 1, j - 1] * m + grad[i - 1, j] * (1 - m)
				else:
					if m > 0:
						gradP1 = grad[i + 1, j - 1] * m + grad[i, j - 1] * (1 - m)
						gradP2 = grad[i - 1, j + 1] * m + grad[i, j + 1] * (1 - m)
					else:
						gradP1 = grad[i - 1, j - 1] * m + grad[i, j - 1] * (1 - m)
						gradP2 = grad[i + 1, j + 1] * m + grad[i, j + 1] * (1 - m)
				if(grad[i,j] >= gradP1 and grad[i,j] >= gradP2):
					# Valid edge
					self.img[i, j] = 255/max*grad[i,j]
				else:
					self.img[i,j] = 0

		for i in range(1, h - 1):
			for j in range(1, w - 1):
				if (self.img[i, j] > tl and (self.img[i - 1, j - 1] > th or self.img[i - 1, j] > th or self.img[i - 1, j + 1] > th or self.img[i, j - 1] > th or self.img[i, j + 1] > th or self.img[i + 1, j - 1] > th or self.img[i + 1, j] > th or self.img[i + 1, j] > th)):
					self.img[i, j] = 128
				if (self.img[i, j] > th):
					self.img[i, j] = 255
				#if (grad[i, j] > tl and (grad[i - 1, j - 1] > th or grad[i - 1, j] > th or grad[i - 1, j + 1] > th or grad[i, j - 1] > th or grad[i, j + 1] > th or grad[i + 1, j - 1] > th or grad[i + 1, j] > th or grad[i + 1, j] > th)):
				#	self.img[i, j] = 128
				#if (grad[i, j] > th):
				#	self.img[i, j] = 255

if __name__ == "__main__":
	Main(sys.argv[1:])