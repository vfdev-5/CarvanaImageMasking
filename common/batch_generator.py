from __future__ import absolute_import

import os
import sys

import numpy as np

# Local repos:
local_repos_path = os.path.abspath(os.path.dirname(__file__))

keras_contrib_path = os.path.join(local_repos_path, "..", "KerasContrib")
if keras_contrib_path not in sys.path:
    sys.path.append(keras_contrib_path)

imgaug_contrib_path = os.path.join(local_repos_path, "..", "imgaug")
if imgaug_contrib_path not in sys.path:
    sys.path.append(imgaug_contrib_path)

from keras_contrib.preprocessing.image.generators import ImageMaskGenerator
# from imgaug import augmenters as iaa

from .data_utils import GENERATED_DATA
from .xy_providers import XYProvider


# def get_imgaug_seq(seed):
#     determinist = {
#         "deterministic": False,
#         "random_state": seed
#     }
#     train_seq = iaa.Sequential([
#         iaa.Sometimes(0.45, iaa.ContrastNormalization(alpha=(0.75, 1.15), **determinist), **determinist),
#         iaa.Add(value=(-35, 35), per_channel=True),
#     ],
#         random_order=True,
#         **determinist
#     )
#     return train_seq
#
#
# def get_id_imgaug_seq():
#     return iaa.Sequential()


class BatchGenerator:
    """
        Batch generator on custom dataset with random data augmentation and data standardization

        Usage:
        ```

        ```
    """

    TRAINING = 'training'
    VALIDATION = 'validation'
    INFERENCE = 'inference'

    def __init__(self, batch_size,
                 xy_provider,
                 random_transform=None,
                 standardization=None,
                 phase=TRAINING,
                 seed=None,
                 verbose=0):

        assert isinstance(xy_provider, XYProvider), "xy_provider should be instance of XYProvider"

        self.batch_size = batch_size
        self.xy_provider = xy_provider
        self.seed = seed
        self.verbose = verbose
        self.phase = phase

    def __call__(self, data_ids):
        pass

        # normalize_data = params.get('normalize_data')
        # normalization = params.get('normalization')
        # save_prefix = params.get('save_prefix')
        # imgaug_seq = params.get('imgaug_seq')

        # assert normalize_data is not None, "normalize_data is needed"
        # assert normalization is not None, "normalization is needed"
        # if normalize_data and (normalization == '' or normalization == 'from_save_prefix'):
        #     assert save_prefix is not None, "save_prefix is needed"
        # assert 'image_size' in params, "image_size is needed"

        # def _random_imgaug(x, y):
        #     x = imgaug_seq.augment_image(255.0 * x) / 255.0
        #     return x, y

        pipeline = ('random_transform', )
        # if imgaug_seq is not None:
        #     pipeline += (_random_imgaug, )
        pipeline += ('standardize', )

        normalize_data = False
        normalization = ''
        save_prefix = None

        random_transform_config = {
            'featurewise_center': normalize_data,
            'featurewise_std_normalization': normalize_data,
            'rotation_range': 45,
            'width_shift_range': 0.15,
            'height_shift_range': 0.15,
            'zoom_range': [0.85, 1.05],
            'horizontal_flip': True,
            'vertical_flip': True,
            'fill_mode': 'nearest'
        }

        gen = ImageMaskGenerator(pipeline=pipeline, **random_transform_config)

        if normalize_data:
            if normalization == '':
                print("\n-- Fit stats of train dataset")
                gen.fit(xy_provider(data_ids, **xy_provider_config),
                        len(data_ids),
                        augment=True,
                        seed=seed,
                        save_to_dir=GENERATED_DATA,
                        save_prefix=save_prefix,
                        batch_size=4,
                        verbose=verbose)
            elif normalization == 'inception' or normalization == 'xception':
                # Preprocessing of Xception: keras/applications/xception.py
                if verbose > 0:
                    print("Image normalization: ", normalization)
                gen.mean = 0.5
                gen.std = 0.5
            elif normalization == 'resnet' or normalization == 'vgg':
                if verbose > 0:
                    print("Image normalization: ", normalization)
                gen.std = 1.0 / 255.0  # Rescale to [0.0, 255.0]
                m = np.array([123.68, 116.779, 103.939]) / 255.0  # RGB
                # if channels_first:
                #     m = m[:, None, None]
                # else:
                m = m[None, None, :]
                gen.mean = m
            elif normalization == 'from_save_prefix':
                assert len(save_prefix) > 0, "WTF"
                # Load mean, std, principal_components if file exists
                filename = os.path.join(GENERATED_DATA, save_prefix + "_stats.npz")
                assert os.path.exists(filename), "WTF"
                if verbose > 0:
                    print("Load existing file: %s" % filename)
                npzfile = np.load(filename)
                gen.mean = npzfile['mean']
                gen.std = npzfile['std']

        # Ensure that all batches have the same size in training phase
        test_mode = self.phase in (BatchGenerator.VALIDATION or BatchGenerator.INFERENCE)
        with_y = self.phase != BatchGenerator.INFERENCE
        ll = len(data_ids) if test_mode else (len(data_ids) // self.batch_size) * self.batch_size
        flow = gen.flow(self.xy_provider(data_ids, test_mode=test_mode, with_y=with_y),
                        ll,
                        seed=self.seed,
                        batch_size=self.batch_size)
        return flow
