import numpy, math

def kernel(name, k = 3):
	{
		'sobelx': numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8,
		'sobelx': numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/8,
		'gauss' : gauss_kernel(k)
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
	row = numpy.array([gaussian(x, kernel_radius, sigma) for x in range(2 * kernel_radius + 1)])
	kernel = row.reshape(-1, 1).dot(row.reshape(1, -1))
	return kernel / numpy.sum(kernel)
	
def dilatate(img):
	h,w,c = img.shape
	for i in range(1,h-1):
		for j in range(1,w-1):
			img[i,j,:] = numpy.max(img[i-1:i+1,j-1:j+1,:])	
	return img

def erode(img):
	h,w,c = img.shape
	for i in range(1,h-1):
		for j in range(1,w-1):
			img[i,j,:] = numpy.min(img[i-1:i+1,j-1:j+1,:])			
	return img
		
def closing(img):
	return erode(dilatate(img))

def opening(img):
	return dilatate(erode(img))
