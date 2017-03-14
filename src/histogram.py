import cv2, numpy, matplotlib.pyplot as pyplot


def build(img, bins=''):
	bins = int(bins or input('Insert number of bins: '))
	h, w, c = img.shape

	return numpy.asarray([x for channel in range(c) for x in numpy.unique(img[:,:,channel] // (256 / bins), return_counts=True)[1] / (h * w)])


def display(img, histogram):
	h, w, c = img.shape

	if(c == 1):
		img = img[:,:,0]
		cmap = 'gray'
	elif(c == 3):
		img = cv2.cvtColor(img.astype(numpy.uint8), cv2.COLOR_BGR2RGB)  # for visualization
		cmap = None
	else:
		raise Exception('Image has weird number of dimensions: %d' % img.ndim)

	assert histogram.size % c == 0, 'Histogram bins are not a multiple of image channels. I consider this wrong.'
	n_bins = histogram.size // c
	x_bar = numpy.tile((numpy.arange(n_bins) * 256 / n_bins), reps=(c,))

	f, axarr = pyplot.subplots(2, 1)
	axarr[0].imshow(img, cmap=cmap)
	axarr[1].bar(numpy.arange(histogram.size), histogram)
	axarr[1].set_xticks(numpy.arange(histogram.size))
	axarr[1].set_xticklabels(x_bar)

	pyplot.show()