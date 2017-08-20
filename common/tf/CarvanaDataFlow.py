
import numpy as np
import cv2

from common.data_utils import train_ids, test_ids
from common.image_utils import _get_image_data_opencv, _get_image_data_pil

from tensorpack import RNGDataFlow


class CarvanaDataFlow(RNGDataFlow):
    """
    Tensorpack flavored Carvana dataset
    """

    def __init__(self, data_ids, with_y=True, shuffle=False, image_size=None, max_n_samples=None):

        assert isinstance(data_ids, list) or isinstance(data_ids, tuple), ""

        self.shuffle = shuffle
        self.data_ids = data_ids
        self.with_y = with_y

        if max_n_samples is not None:
            self.data_ids = self.data_ids[:max_n_samples]

        self.image_size = image_size

    def size(self):
        return len(self.data_ids)

    @staticmethod
    def _to_mask_id(img_id):
        mask_id = list(img_id)
        mask_id[1] += '_mask'
        return mask_id

    def get_data(self):

        def _resize(img):
            return cv2.resize(img, dsize=self.image_size)

        _resize_func = _resize if self.image_size is not None else \
            lambda img: img

        idxs = np.arange(len(self.data_ids))
        if self.shuffle:
            self.rng.shuffle(idxs)

        for index in idxs:
            img_id = self.data_ids[index]
            mask_id = CarvanaDataFlow._to_mask_id(img_id)

            img = _get_image_data_opencv(*img_id)
            img = _resize_func(img)

            if self.with_y:
                mask = _get_image_data_pil(*mask_id)
                mask = _resize_func(mask)
                mask = mask[:, :, np.newaxis]
            else:
                mask = None

            yield [img, mask, img_id]





