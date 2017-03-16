import numpy, math

def convolution(img, kernel):

	kernel = numpy.array(kernel)

	rowk = kernel[0]
	colk = kernel.transpose()[0] / kernel[0,0]

	if not (abs(colk.reshape(-1,1).dot(rowk.reshape(1,-1)) - kernel) < exp(-15)).all():
		print("Kernel not symmetric")
		return None

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

def gaussian(x, mu, sigma):
	return math.exp( -(((x-mu)/(sigma))**2)/2.0 )

def gauss_kernel(kernel_radius):
	sigma = kernel_radius / 2.
	row = numpy.array([gaussian(x, kernel_radius, sigma) for x in range(2 * kernel_radius + 1)])
	kernel = row.reshape(-1, 1).dot(row.reshape(1, -1))
	return kernel / numpy.sum(kernel)