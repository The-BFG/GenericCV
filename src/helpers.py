import numpy as np
import cv2


def connected_components(img_bin, verbose=False):
    """
    Helper function to get the binary masks of different
    objects given a binary image.

    :param img_bin: binary image containing objects to segment.
    :param verbose: if True, shows segmentation.
    :return: a tuple of lists, like (masks, contours)
    """
    _, contours, hierarchy = cv2.findContours(img_bin, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]

    # find border contour
    border = np.argmax(np.array([c.shape[0] for c in contours]))

    # find external
    external_contours = []
    first_external = hierarchy[border, 2]
    cur_cont = first_external
    while cur_cont != -1:
        external_contours.append(cur_cont)
        cur_cont = hierarchy[cur_cont, 0]  # next contour at the same hierarchical level

    ret_masks = []
    for ext_c in external_contours:
        mask = np.zeros_like(img_bin)
        cv2.drawContours(mask, contours=contours, contourIdx=ext_c, color=255, thickness=cv2.FILLED)

        # find cc at same level of background
        cur_back = hierarchy[border, 1]
        while cur_back != -1:
            cv2.drawContours(mask, contours=contours, contourIdx=cur_back, color=0, thickness=cv2.FILLED)
            cur_back = hierarchy[cur_back, 1]  # same hierarchical level

        if verbose:
            cv2.imshow('', mask)
            cv2.waitKey()

        ret_masks.append(mask)

    return ret_masks, [contours[i] for i in external_contours]

class_dict = {0: 'conrod', 1: 'washer', 2: 'screw'}


def show_predictions(masks, predictions, timeout=0):
    """
    Function to plot predictions over a binary mask.

    :param masks: A list binary mask having shape (h,w).
    :param predictions: A list of predictions
    :param timeout: optional timeout for visualization
    """

    for mask, pred in zip(masks, predictions):

        idx = np.where(mask != 0)

        y_min = np.min(idx[0])
        y_max = np.max(idx[0])
        x_min = np.min(idx[1])
        x_max = np.max(idx[1])

        to_show = np.tile(np.expand_dims(mask, axis=2), reps=(1, 1, 3))
        cv2.rectangle(to_show, pt1=(x_min, y_min), pt2=(x_max, y_max), color=(0, 255, 0), thickness=3)
        cv2.putText(to_show, class_dict[int(pred)], (x_min, y_min-5), 3, 0.5, color=(0, 255, 0))

        cv2.imshow('Prediction', to_show)
        cv2.waitKey(timeout)


def read_lines_from_text_file(filename):
    """
    Helper function to return lines of a text file as a list of strings.

    :param filename: the file to read
    :return: a list of strings, one per line
    """

    with open(filename) as f:
        content = f.readlines()

    return content
