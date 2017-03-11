import numpy

def convolution(img):
	mat = [numpy.convolve(numpy.squeeze(row), [1 / 4, 1 / 2, 1 / 4], 'valid') for row in img]
	img = numpy.transpose(numpy.array(mat))
	mat = [numpy.convolve(numpy.squeeze(row), [1 / 4, 1 / 2, 1 / 4], 'valid') for row in img]
	img = numpy.transpose(numpy.array(mat))
	return numpy.expand_dims(img, axis=2).astype(numpy.uint8)

