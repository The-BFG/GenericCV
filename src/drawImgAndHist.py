import numpy as np
import cv2

import matplotlib.pyplot as plt


def drawImgAndHist(img, histogram):
	"""
	Draws an image and its histogram in cool subplots.

	:param img: The image to visualize. Numpy array having shape (h,w,c) or (h,w)
	:param histogram: the image histogram. Numpy array having shape (c*n_bins)
	:return: None
	"""

	if img.ndim == 3:
		h, w, c = img.shape
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for visualization
		cmap = None
	elif img.ndim == 2:
		h, w, c = img.shape + (1,)
		cmap = 'gray'
	else:
		raise Exception('img has weird number of dimensions: {}'.format(img.ndim))

	assert histogram.size % c == 0, 'Histogram bins are not a multiple of image channels. I consider this wrong.'
	n_bins = histogram.size // c
	x_bar = np.tile((np.arange(n_bins) * 256 / n_bins), reps=(c,))

	f, axarr = plt.subplots(2, 1)
	axarr[0].imshow(img, cmap=cmap)
	axarr[1].bar(np.arange(histogram.size), histogram)
	axarr[1].set_xticks(np.arange(histogram.size))
	axarr[1].set_xticklabels(x_bar)

	plt.show()
