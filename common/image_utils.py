from __future__ import absolute_import

import cv2
from PIL import Image
import numpy as np

# Project
from .data_utils import get_filename


def get_image_data(image_id, image_type, **kwargs):
    """
    Method to get image data as np.array specifying image id and type
    """
    cache = kwargs.get('cache')
    if cache is not None:
        key = (image_id, image_type)
        if key in cache:
            return cache.get(key)

    if 'mask' in image_type:
        img = _get_image_data_pil(image_id, image_type, **kwargs)
    else:
        img = _get_image_data_opencv(image_id, image_type, **kwargs)

    if cache is not None:
        key = (image_id, image_type)
        cache.put(key, img)
    return img


def imwrite(img, image_id, image_type):
    output_filename = get_filename(image_id, image_type)
    if '...' in image_type:
        np.savez_compressed(output_filename, img)
    else:
        pil_image = Image.fromarray(img)
        pil_image.save(output_filename)


def _get_image_data_opencv(image_id, image_type, **kwargs):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s (%s)" % (image_id, image_type, fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _get_image_data_pil(image_id, image_type, return_exif_md=False, return_shape_only=False, **kwargs):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    try:
        img_pil = Image.open(fname)
    except Exception as e:
        assert False, "Failed to read image : %s, %s. Error message: %s" % (image_id, image_type, e)

    if return_shape_only:
        return img_pil.size[::-1] + (len(img_pil.getbands()),)

    img = np.asarray(img_pil)
    assert isinstance(img, np.ndarray), "Open image is not an ndarray. Image id/type : %s, %s" % (image_id, image_type)
    if not return_exif_md:
        return img
    else:
        return img, img_pil._getexif()


def median_blur(img, ksize):
    init_shape = img.shape
    img2 = np.expand_dims(img, axis=2) if len(init_shape) == 2 else img
    out = np.zeros_like(img2)
    img_n, mins, maxs = normalize(img2, return_mins_maxs=True)
    for i in range(img2.shape[2]):
        img_temp = (255.0 * img_n[:, :, i]).astype(np.uint8)
        img_temp = 1.0 / 255.0 * cv2.medianBlur(img_temp, ksize).astype(img.dtype)
        out[:, :, i] = maxs[i] * img_temp + mins[i]
    out = out.reshape(init_shape)
    return out


def spot_cleaning(img, ksize, threshold=0.15):
    """
    ksize : kernel size for median blur
    threshold for outliers, [0.0,1.0]
    https://github.com/kmader/Quantitative-Big-Imaging-2016/blob/master/Lectures/02-Slides.pdf
    """
    init_type = img.dtype
    init_shape = img.shape
    if len(init_shape) == 2:
        img = img[:, :, None]
    img_median = median_blur(img, ksize).astype(np.float32)
    diff = np.abs(img.astype(np.float32) - img_median)
    diff = np.mean(diff, axis=2)
    diff = normalize(diff, q_min=0, q_max=100)
    diff2 = diff.copy()
    _, diff = cv2.threshold(diff, threshold, 1.0, cv2.THRESH_BINARY)
    _, diff2 = cv2.threshold(diff2, threshold, 1.0, cv2.THRESH_BINARY_INV)

    img_median2 = img_median * diff[:, :, None]
    img2 = img * diff2[:, :, None]
    img2 += img_median2
    if img2.shape != init_shape:
        img2 = img2.reshape(init_shape)
    return img2.astype(init_type)


def normalize(in_img, q_min=0.5, q_max=99.5, return_mins_maxs=False):
    """
    Normalize image in [0.0, 1.0]
    mins is array of minima
    maxs is array of differences between maxima and minima
    """
    init_shape = in_img.shape
    if len(init_shape) == 2:
        in_img = np.expand_dims(in_img, axis=2)
    w, h, d = in_img.shape
    img = in_img.copy()
    img = np.reshape(img, [w * h, d]).astype(np.float64)
    mins = np.percentile(img, q_min, axis=0)
    maxs = np.percentile(img, q_max, axis=0) - mins
    maxs[(maxs < 0.0001) & (maxs > -0.0001)] = 0.0001
    img = (img - mins[None, :]) / maxs[None, :]
    img = img.clip(0.0, 1.0)
    img = np.reshape(img, [w, h, d])
    if init_shape != img.shape:
        img = img.reshape(init_shape)
    if return_mins_maxs:
        return img, mins, maxs
    return img


def scale_percentile(matrix, q_min=0.5, q_max=99.5):
    is_gray = False
    if len(matrix.shape) == 2:
        is_gray = True
        matrix = matrix.reshape(matrix.shape + (1,))
    matrix = (255 * normalize(matrix, q_min, q_max)).astype(np.uint8)
    if is_gray:
        matrix = matrix.reshape(matrix.shape[:2])
    return matrix
