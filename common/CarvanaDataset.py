import os

from PIL import Image

from torch import is_tensor, FloatTensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .data_utils import train_ids, test_ids, get_filename
from .image_utils import _get_image_data_opencv


def get_pil_image_data(image_id, image_type, **kwargs):
    fname = get_filename(image_id, image_type)
    try:
        img_pil = Image.open(fname)
    except Exception as e:
        assert False, "Failed to read image : %s, %s. Error message: %s" % (image_id, image_type, e)
    return img_pil


class CarvanaDataset(Dataset):
    """
        PyTorch Carvana images Dataset
    """
    TRAINVAL = 0
    TEST = 1

    def __init__(self, dataset_type, geom_transform=None, color_transform=None):
        self.dataset_type = dataset_type

        if self.dataset_type == CarvanaDataset.TRAINVAL:
            self.data_ids = [(data_id, "Train") for data_id in train_ids]
        elif self.dataset_type == CarvanaDataset.TEST:
            self.data_ids = [(data_id, "Test") for data_id in test_ids]
        self.geom_transform = geom_transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.data_ids)

    @staticmethod
    def _to_mask_id(img_id):
        mask_id = list(img_id)
        mask_id[1] += '_mask'
        return mask_id



class CarvanaDatasetPIL(CarvanaDataset):
    """
        PyTorch Carvana images Dataset based on PIL
    """
    def __getitem__(self, index):
        img_id = self.data_ids[index]
        mask_id = CarvanaDataset._to_mask_id(img_id)
        img = get_pil_image_data(*img_id)
        mask = get_pil_image_data(*mask_id)

        if self.color_transform is not None:
            img = self.color_transform(img)

        if self.geom_transform is not None:
            pass

        if not is_tensor(img):
            img = ToTensor()(img)

        if not is_tensor(mask):
            mask = ToTensor()(mask)

        return img, mask


# class CarvanaDatasetCV2(CarvanaDataset):
#     """
#         PyTorch Carvana images Dataset based on OpenCV
#     """
#     def __getitem__(self, index):
#         img_id = self.data_ids[index]
#         mask_id = CarvanaDataset._to_mask_id(img_id)
#         img = _get_image_data_opencv(*img_id)
#         mask = _get_image_data_opencv(*mask_id)
#
#         if self.color_transform is not None:
#             img = self.color_transform(img)
#
#         if self.geom_transform is not None:
#             pass
#
#         if not is_tensor(img):
#             img = FloatTensor(img)
#
#         if not is_tensor(mask):
#             mask = FloatTensor(mask)
#
#         return img, mask
