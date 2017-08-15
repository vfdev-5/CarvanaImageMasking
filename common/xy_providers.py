
import numpy as np


class XYProvider:
    """
        Interface to custom dataset

        Usage:
        ```
        X = [img0, img1, img2, ...]
        Y = [label0, label1, label2, ...]

        def get_x_data(i, **kwargs):
            return X[i]

        def get_y_data(i, **kwargs):
            return Y[i]

        data_ids = [(0,), (1,), (2,), (3,), ...]

        xy_gen = XYProvider(get_x_data=get_x_data, get_y_data=get_y_data)
        # Infinite data generator for training dataset:
        for x, y, _ in xy_gen(data_ids, test_mode=False, with_y=True):
            print(x.shape, y)
        ```
    """

    def __init__(self, get_x_data, get_y_data=None, seed=None):
        """
        Initialize custom dataset generator

        :param get_x_data: function to get x data from a data id
        :param get_y_data: function to get y data from a data id
        """
        assert callable(get_x_data), "get_x_data should be callable"
        if get_y_data is not None:
            assert callable(get_y_data), "get_y_data should be callable"

        self.get_x_data = get_x_data
        self.get_y_data = get_y_data
        self.seed = seed

    def __call__(self, data_ids, test_mode=False, with_y=True):
        """
        Data generator from a list of data identifiers

        :param data_ids: list of data identifiers such that each item should be accepted by `get_x_data` and `get_y_data`
        functions (see kwargs). For example `get_x_data(*data_ids[0])` should return x data corresponding to `data_ids[0]`.
        :param test_mode: infinite (False) or finite (True) data generator.
        :param with_y: returns y data if True, otherwise None.
        :return: x and y data and data id
        """
        assert isinstance(data_ids, list) or isinstance(data_ids, tuple), "Argument data_ids should be a list or tuple"
        if with_y:
            assert self.get_y_data is not None, "get_y_data should not be None"

        if self.seed is not None and not test_mode:
            np.random.seed(self.seed)

        if not test_mode:
            data_ids = list(data_ids)

        while True:
            if not test_mode:
                np.random.shuffle(data_ids)
            for i, data_id in enumerate(data_ids):
                x_data = self.get_x_data(*data_id)

                if with_y:
                    y_data = self.get_y_data(*data_id)
                else:
                    y_data = None

                yield x_data, y_data, data_id

            if test_mode:
                return


# import numpy as np
# import cv2
#
# from image_utils import get_image_data
#
#
# # def image_class_labels_provider(image_id_type_list,
# #                                 image_size,
# #                                 class_index,
# #                                 channels_first=True,
# #                                 test_mode=False,
# #                                 seed=None,
# #                                 cache=None,
# #                                 verbose=0, **kwargs):
# #
# #     if seed is not None:
# #         np.random.seed(seed)
# #
# #     counter = 0
# #     image_id_type_list = list(image_id_type_list)
# #     while True:
# #         np.random.shuffle(image_id_type_list)
# #         for i, (image_id, image_type) in enumerate(image_id_type_list):
# #             if verbose > 0:
# #                 print("Image id/type:", image_id, image_type, "| counter=", i)
# #
# #             key = (image_id, image_type)
# #             if cache is not None and key in cache:
# #                 if verbose > 0:
# #                     print("-- Load from RAM")
# #                 img, label = cache.get(key)
# #
# #                 if channels_first:
# #                     if img.shape[1:] != image_size[::-1]:
# #                         img = img.transpose([1, 2, 0])
# #                         img = cv2.resize(img, dsize=image_size[::-1])
# #                         img = img.transpose([2, 0, 1])
# #                 else:
# #                     if img.shape[:2] != image_size[::-1]:
# #                         img = cv2.resize(img, dsize=image_size[::-1])
# #             else:
# #                 if verbose > 0:
# #                     print("-- Load from disk")
# #
# #                 img = get_image_data(image_id, image_type)
# #
# #                 if img.shape[:2] != image_size:
# #                     img = cv2.resize(img, dsize=image_size)
# #                 if channels_first:
# #                     img = img.transpose([2, 0, 1])
# #
# #                 img = img.astype(np.float32) / 255.0
# #
# #                 if class_index is not None:
# #                     label = get_label(image_id, image_type, class_index=class_index)
# #                 else:
# #                     label = None
# #                 # fill the cache only at first time:
# #                 if cache is not None and counter == 0:
# #                     cache.put(key, (img, label))
# #
# #             if test_mode:
# #                 yield img, label, (image_id, image_type)
# #             else:
# #                 yield img, label
# #
# #         if test_mode:
# #             return
# #         counter += 1
# #
# #
# # def to_ndwi(img):
# #     """
# #     NDWI = (Xgreen – Xnir)/(Xgreen + Xnir)
# #     """
# #     b, g, r, ir = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
# #     g = g.astype(np.float32)
# #     ir = ir.astype(np.float32)
# #     return (g - ir) / (g + ir + 1e-10)
# #
# #
# # def to_lightness(img):
# #     img32 = img.astype(np.float32)
# #     hls32 = cv2.cvtColor(img32, cv2.COLOR_RGB2HLS)
# #     l32 = hls32[:, :, 1]
# #     return l32
# #
# # def to_ndvi(img):
# #     """
# #     NDVI = (Nir - r) / (Nir + r)
# #     """
# #     b, g, r, ir = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]
# #     r = r.astype(np.float32)
# #     ir = ir.astype(np.float32)
# #     return (ir - r) / (r + ir + 1e-10)
# #
# #
# # def tif_image_label_provider(image_id_type_list,
# #                              image_size,
# #                              channels_first=True,
# #                              test_mode=False,
# #                              seed=None,
# #                              cache=None,
# #                              with_label=True,
# #                              class_index=None,
# #                              tag=None,
# #                              verbose=0, **kwargs):
# #
# #     assert not (class_index is not None and tag is not None), "Either class_index or either tag, not both"
# #
# #     if seed is not None:
# #         np.random.seed(seed)
# #
# #     counter = 0
# #     image_id_type_list = list(image_id_type_list)
# #     while True:
# #         np.random.shuffle(image_id_type_list)
# #         for i, (image_id, image_type) in enumerate(image_id_type_list):
# #             if verbose > 0:
# #                 print("Image id/type:", image_id, image_type, "| counter=", i)
# #
# #             key = (image_id, image_type)
# #             if cache is not None and key in cache:
# #                 if verbose > 0:
# #                     print("-- Load from RAM")
# #                 img, label = cache.get(key)
# #
# #                 if channels_first:
# #                     if img.shape[1:] != image_size[::-1]:
# #                         img = img.transpose([1, 2, 0])
# #                         img = cv2.resize(img, dsize=image_size[::-1])
# #                         img = img.transpose([2, 0, 1])
# #                 else:
# #                     if img.shape[:2] != image_size[::-1]:
# #                         img = cv2.resize(img, dsize=image_size[::-1])
# #             else:
# #
# #                 if verbose > 0:
# #                     print("-- Load from disk")
# #
# #                 tif_img = get_image_data(image_id, image_type)
# #
# #                 if tif_img.shape[:2] != image_size:
# #                     tif_img = cv2.resize(tif_img, dsize=image_size)
# #                 if channels_first:
# #                     tif_img = tif_img.transpose([2, 0, 1])
# #
# #                 tif_img = tif_img.astype(np.float32) / 255.0
# #
# #                 # [rgb + nir (originasl tif)] + ndvi + ndwi + lightness
# #                 img = np.zeros(image_size + (7, ), dtype=np.float32)
# #                 img[:, :, :4] = tif_img
# #                 img[:, :, 4] = to_ndvi(tif_img)
# #                 img[:, :, 5] = to_ndwi(tif_img)
# #                 img[:, :, 6] = to_lightness(tif_img)
# #
# #                 if with_label:
# #                     if class_index is not None:
# #                         label = get_label(image_id, image_type, class_index=class_index)
# #                         label = np.concatenate((label, [0, ]))
# #                         if np.sum(label) < 1:
# #                             label[-1] = 1
# #                     elif tag is not None:
# #                         label = get_label(image_id, image_type, tag=tag)
# #                     else:
# #                         label = get_label(image_id, image_type)
# #
# #                 else:
# #                     label = None
# #                 # fill the cache only at first time:
# #                 if cache is not None and counter == 0:
# #                     cache.put(key, (img, label))
# #
# #             if test_mode:
# #                 yield img, label, (image_id, image_type)
# #             else:
# #                 yield img, label
# #
# #         if test_mode:
# #             return
# #         counter += 1
# #
# #
# # def image_label_provider(image_id_type_list,
# #                          image_size,
# #                          channels_first=True,
# #                          test_mode=False,
# #                          seed=None,
# #                          cache=None,
# #                          with_label=True,
# #                          verbose=0, **kwargs):
# #
# #     if seed is not None:
# #         np.random.seed(seed)
# #
# #     counter = 0
# #     image_id_type_list = list(image_id_type_list)
# #     while True:
# #         np.random.shuffle(image_id_type_list)
# #         for i, (image_id, image_type) in enumerate(image_id_type_list):
# #             if verbose > 0:
# #                 print("Image id/type:", image_id, image_type, "| counter=", i)
# #
# #             key = (image_id, image_type)
# #             if cache is not None and key in cache:
# #                 if verbose > 0:
# #                     print("-- Load from RAM")
# #                 img, label = cache.get(key)
# #
# #                 if channels_first:
# #                     if img.shape[1:] != image_size[::-1]:
# #                         img = img.transpose([1, 2, 0])
# #                         img = cv2.resize(img, dsize=image_size[::-1])
# #                         img = img.transpose([2, 0, 1])
# #                 else:
# #                     if img.shape[:2] != image_size[::-1]:
# #                         img = cv2.resize(img, dsize=image_size[::-1])
# #             else:
# #
# #                 if verbose > 0:
# #                     print("-- Load from disk")
# #
# #                 img = get_image_data(image_id, image_type)
# #
# #                 if img.shape[:2] != image_size:
# #                     img = cv2.resize(img, dsize=image_size)
# #                 if channels_first:
# #                     img = img.transpose([2, 0, 1])
# #
# #                 img = img.astype(np.float32) / 255.0
# #
# #                 if with_label:
# #                     label = get_label(image_id, image_type)
# #                 else:
# #                     label = None
# #                 # fill the cache only at first time:
# #                 if cache is not None and counter == 0:
# #                     cache.put(key, (img, label))
# #
# #             if test_mode:
# #                 yield img, label, (image_id, image_type)
# #             else:
# #                 yield img, label
# #
# #         if test_mode:
# #             return
# #         counter += 1
#
#
# def image_mask_provider(image_id_type_list,
#                         image_size=(224, 224),
#                         channels_first=True,
#                         test_mode=False,
#                         seed=None,
#                         with_mask=True,
#                         cache=None,
#                         verbose=0, **kwargs):
#     if seed is not None:
#         np.random.seed(seed)
#
#     def _resize(_img, _image_size, _channels_first, _interpolation):
#         if _channels_first:
#             if _img.shape[1:] != _image_size[::-1]:
#                 _img = _img.transpose([1, 2, 0])
#                 _img = cv2.resize(_img, dsize=_image_size[::-1], interpolation=_interpolation)
#                 _img = _img.transpose([2, 0, 1])
#         else:
#             if _img.shape[:2] != _image_size[::-1]:
#                 _img = cv2.resize(_img, dsize=_image_size[::-1], interpolation=_interpolation)
#         return _img
#
#     counter = 0
#     image_id_type_list = list(image_id_type_list)
#     while True:
#         np.random.shuffle(image_id_type_list)
#         for i, (image_id, image_type) in enumerate(image_id_type_list):
#             if verbose > 0:
#                 print("Image id/type:", image_id, image_type, "| counter=", i)
#
#             key = (image_id, image_type)
#             if cache is not None and key in cache:
#                 if verbose > 0:
#                     print("-- Load from RAM")
#                 img, mask = cache.get(key)
#
#                 img = _resize(img, image_size, False, cv2.INTER_CUBIC)
#                 mask = _resize(mask, image_size, False, cv2.INTER_LINEAR)
#
#             else:
#                 if verbose > 0:
#                     print("-- Load from disk")
#
#                 img = get_image_data(image_id, image_type)
#
#                 img = _resize(img, image_size, False, cv2.INTER_CUBIC)
#                 if channels_first:
#                     img = img.transpose([2, 0, 1])
#                 img = img.astype(np.float32) / 255.0
#
#                 if with_mask:
#                     mask = get_image_data(image_id, "Train_mask")
#                     mask = _resize(mask, image_size, False, cv2.INTER_LINEAR)
#                     # Mask should be binary
#                     if mask.max() > 1:
#                         mask = (mask > 1).astype(np.unit8)
#                     mask = mask[:, :, None]
#                     if channels_first:
#                         mask = mask.transpose([2, 0, 1])
#                 else:
#                     mask = None
#
#                 # fill the cache only at first time:
#                 if cache is not None and counter == 0:
#                     cache.put(key, (img, mask))
#
#             if test_mode:
#                 yield img, mask, (image_id, image_type)
#             else:
#                 yield img, mask
#
#         if test_mode:
#             return
#         counter += 1
#
