#!/usr/bin/python3
import sys, getopt, cv2, numpy
import histogram

from linearPointOperators import *
from neighborhoodOperators import *

class Main:

	def __init__(self, argv):

		self.path, self.loadType, self.commands, self.histogram = None, None, None, None

		# TODO: Aggiungere opzione "-c commandi:..."
		# TODO: Aggiungere opzione per fare solo chiamate cv2
		self.options = {"-i": "self.path", "-t": "self.loadType", "-c": "self.commands"}

		self.menu = {
			'0': ("Exit program", sys.exit),
			'1': ("Image", {
				'a': ('Load image', self.load),
				'b': ('Show image', self.show),
				'c': ('Save image', self.save)
			}),
			'2': ("Linear point operations", {
				'a': ("Generate the image histogram", self.createHistogram),
				'b': ("Change contrast and brightness", self.changeContrastBrightness),
				'c': ("Apply contrast stretching", self.stretchContrast),
				'd': ("Make a threshold", self.makeThreshold),
				'e': ("Otzu threashold on the image", self.makeOtsu),
			}),
			'3': ('Neighborhood operations', {
				'a': ("Convolution", self.performConvolution),
				'b': ("Sobel", self.sobel),
				'c': ("Canny", self.canny),
				'd': ("Segmentation", self.segmentation),
				'd': ("Morphology", self.morphology),
				'e': ("Labeling", self.labeling)
			}),
			'4': ('Transformations', {
				'a': ("Translation", self.translation),
				'b': ("Scaling", self.translation),
				'c': ("Rotation", self.translation),
				'd': ("Projection", self.translation),

			})
		}

		try:
			opts, args = getopt.getopt(argv, "i:t:c:")
		except getopt.GetoptError:
			print('main.py -i <inputfile> -t <loadtype> -c <commands>')
			sys.exit(2)
		for opt, arg in opts:
			exec(self.options[opt] + ' = arg')

		if self.path:
			self.load(self.path, self.loadType)



		while True:
			print("Choose:\n" + "\n".join("%d) %s" % (int(k), v[0]) for k, v in sorted(self.menu.items(),key=lambda a: int(a[0]) if(int(a[0])>0) else sys.maxsize)))
			self.menu[input()][1]()

	def load(self, path = None, loadType = None):
		self.path = path if path else input('Insert new path: ')
		self.loadType = loadType if loadType else input("Choose image loading type:\n0) Grayscale Image\n1) Color image\n")
		self.img = cv2.imread(self.path, int(self.loadType))
		if self.img.ndim == 2:
			self.img = numpy.expand_dims(self.img, axis=2)

	def show(self):

		cv2.namedWindow('Image', cv2.WINDOW_FREERATIO)
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

	def stretchContrast(self):
		self.img = contrastStreching(self.img)

	def makeThreshold(self):
		self.img = threshold(self.img, int(input("Insert the threshold: ")))

	def makeOtsu(self):
		self.histogram = histogram.build(self.img, bins=256)
		self.img = threshold(self.img, otsu(self.histogram))

	def performConvolution(self):
		self.img = convolution(self.img, kernel("gauss",8))

	def sobel(self):
		gradx = convolution(self.img, kernel("sobel"))
		grady = convolution(self.img, kernel("sobel").transpose())
		self.img = numpy.sqrt(gradx ** 2 + grady ** 2).astype(numpy.uint8)
		self.img = contrastStreching(self.img,numpy.min(self.img), numpy.max(self.img), 0, 255)

	# TODO: Not working
	def canny(self):

		self.img = cv2.Canny(self.img,10,200)
		return

		self.img = convolution(self.img, kernel("gauss", 8))

		gradx = convolution(self.img, kernel("sobel"))
		grady = convolution(self.img, kernel("sobel").transpose())
		grad = numpy.sqrt(gradx ** 2 + grady ** 2).astype(numpy.uint8)

		tan = numpy.arctan(grady / gradx)
		h, w, c = grad.shape

		max = numpy.max(grad)
		tl = 0.90 * max
		th = 0.99 * max

		# IBW = Gradiente dopo una soglia
		soglia = (numpy.max(grad)+ numpy.min(grad)) /2
		grad = (grad > soglia)*255

		for i in range(1, h - 1):
			for j in range(1, w - 1):
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
					self.img[i, j] = 255 /max*grad[i,j]
				else:
					self.img[i,j] = 0

		# self.img = grad


	def morphology(self):
		menu = {
			'a': ("Dilatate",dilatate),
			'b': ("Erode", erode),
			'c': ("Opening", opening),
			'd': ("Closing", closing),
			'e': ("Gradient", gradient)
		 }

		print("Which kind of morphology operation do you want to do:\n" + "\n".join("%c) %s" % (k, v[0]) for k, v in sorted(menu.items())))

		self.img = menu[input()][1](self.img)

	def labeling(self):
		pass

if __name__ == "__main__":
	Main(sys.argv[1:])
