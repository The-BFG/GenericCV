import numpy

def convolution(img, kernel):

	kernel = numpy.array(kernel)

	rowk = kernel[0]
	colk = kernel.transpose()[0] / kernel[0,0]

	# mat = [row_convolution(row, [1 / 4, 1 / 2, 1 / 4]) for row in img]

	mat = [row_convolution(row,  rowk) for row in img]
	img = numpy.transpose(numpy.array(mat))

	mat = [row_convolution(row, colk) for row in img]
	img = numpy.transpose(numpy.array(mat))

	return numpy.expand_dims(img, axis=2).astype(numpy.uint8)

def row_convolution(a,b):

	column = numpy.arange(len(a)-len(b)+1).reshape((len(a)-len(b)+1, 1))
	row = numpy.arange(len(b)-1, -1, -1)

	index = column + row

	A = numpy.array(a)[index]

	return numpy.dot(A.reshape(A.shape[:2]), numpy.array(b))
