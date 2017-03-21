import numpy, math

def kernel(name, k = 3):
	return {
		'sobel': numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8,
		'gauss': gauss_kernel(k)
	}.get(name)

def convolution(img, kernel):

	kernel = numpy.array(kernel)

	rowk = kernel[0]
	colk = kernel.transpose()[0] / kernel[0,0]

	if not (abs(colk.reshape(-1,1).dot(rowk.reshape(1,-1)) - kernel) < math.exp(-15)).all():
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

def gauss_kernel(kernel_radius):
	gaussian = lambda x, mu, sigma: math.exp( -(((x-mu)/(sigma))**2)/2.0 )
	sigma = kernel_radius / 2.
	sigma = 1
	row = numpy.array([gaussian(x, kernel_radius, sigma) for x in range(2 * kernel_radius + 1)])
	kernel = row.reshape(-1, 1).dot(row.reshape(1, -1))
	return kernel / numpy.sum(kernel)


def row_dilatate(a):
	column = numpy.arange(len(a) - 2).reshape((len(a) - 2, 1))
	row = numpy.arange(2, -1, -1)

	index = column + row

	A = numpy.array(a)[index]

	return numpy.max(A, axis=1)


def dilatate(img):
	img = img[:, :, 0]

	mat = [row_dilatate(row) for row in img]
	img = numpy.transpose(numpy.array(mat))

	mat = [row_dilatate(row) for row in img]
	img = numpy.transpose(numpy.array(mat))

	return numpy.expand_dims(img, axis=2).astype(numpy.uint8)


def row_erode(a):
	column = numpy.arange(len(a) - 2).reshape((len(a) - 2, 1))
	row = numpy.arange(2, -1, -1)

	index = column + row

	A = numpy.array(a)[index]

	return numpy.min(A, axis=1)


def erode(img):
	img = img[:, :, 0]

	mat = [row_erode(row) for row in img]
	img = numpy.transpose(numpy.array(mat))

	mat = [row_erode(row) for row in img]
	img = numpy.transpose(numpy.array(mat))

	return numpy.expand_dims(img, axis=2).astype(numpy.uint8)

def closing(img):
	return erode(dilatate(img))

def opening(img):
	return dilatate(erode(img))

def gradient(img):
	return dilatate(img) - erode(img)