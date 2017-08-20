
import os
from glob import glob
import pandas as pd

project_common_path = os.path.dirname(__file__)

if 'INPUT_PATH' in os.environ:
    INPUT_PATH = os.environ['INPUT_PATH']
else:
    INPUT_PATH = os.path.join(project_common_path, '..', 'input')

if 'OUTPUT_PATH' in os.environ:
    OUTPUT_PATH = os.environ['OUTPUT_PATH']
else:
    OUTPUT_PATH = os.path.join(project_common_path, '..', 'output')

DATA_PATH = INPUT_PATH
TRAIN_DATA = os.path.join(DATA_PATH, "train")
TRAIN_MASKS_DATA = os.path.join(DATA_PATH, "train_masks")
TEST_DATA = os.path.join(DATA_PATH, "test")
TRAIN_MASKS_CSV_FILEPATH = os.path.join(DATA_PATH, "train_masks.csv")
METADATA_CSV_FILEPATH = os.path.join(DATA_PATH, "metadata.csv")

GENERATED_DATA = os.path.join(OUTPUT_PATH, 'generated')
RESOURCES_PATH = os.path.join(project_common_path, '..', 'resources')

if not os.path.exists(GENERATED_DATA):
    os.makedirs(GENERATED_DATA)

assert os.path.isfile(TRAIN_MASKS_CSV_FILEPATH), "File %s is not found" % TRAIN_MASKS_CSV_FILEPATH
TRAIN_MASKS_CSV = pd.read_csv(TRAIN_MASKS_CSV_FILEPATH)
TRAIN_MASKS_CSV['id'] = TRAIN_MASKS_CSV['img'].apply(lambda x: x[:-7])

assert os.path.isfile(METADATA_CSV_FILEPATH), "File %s is not found" % METADATA_CSV_FILEPATH
METADATA_CSV = pd.read_csv(METADATA_CSV_FILEPATH)
METADATA_CSV.loc[651, 'make'] = 'Chevrolet'
METADATA_CSV.loc[1789, 'make'] = 'Toyota'
METADATA_CSV.loc[2138, 'make'] = 'Volvo'
METADATA_CSV.loc[2373, 'make'] = 'Hyundai'
METADATA_CSV.loc[2400, 'make'] = 'Kia'
METADATA_CSV.loc[2966, 'make'] = 'Nissan'
METADATA_CSV.loc[4027, 'make'] = 'Hyundai'
METADATA_CSV.loc[4270, 'make'] = 'Hyundai'
METADATA_CSV.loc[4630, 'make'] = 'GMC'
METADATA_CSV.loc[5222, 'make'] = 'GMC'
METADATA_CSV.loc[5304, 'make'] = 'Toyota'


train_files = glob(os.path.join(TRAIN_DATA, "*.jpg"))
train_ids = [s[len(TRAIN_DATA)+1:-4] for s in train_files]

if len(train_files) == 0:
    print("No trainined data found at %s " % TRAIN_DATA)


test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = [s[len(TEST_DATA)+1:-4] for s in test_files]

if len(test_ids) == 0:
    print("No test data found at %s " % TEST_DATA)


UNIQUE_CARS = sorted(METADATA_CSV['id'].unique())
UNIQUE_TRAIN_CARS = sorted(list(TRAIN_MASKS_CSV['id']))
UNIQUE_CAR_MAKERS = sorted(METADATA_CSV['make'].unique())
UNIQUE_TRAIN_CAR_MAKERS = sorted(METADATA_CSV[METADATA_CSV['id'].isin(UNIQUE_TRAIN_CARS)]['make'].unique())


def get_trainval_ids_with_makers():
    _metadata_csv = METADATA_CSV.copy()
    _metadata_csv.index = METADATA_CSV['id']

    def get_car_maker(data_id):
        _data_id = data_id[0][:-3]
        return _metadata_csv.loc[_data_id, 'make']

    trainval_ids = [(data_id, "Train") for data_id in train_ids]
    trainval_ids_makers = [get_car_maker(data_ids) for data_ids in trainval_ids]
    return trainval_ids, trainval_ids_makers


def get_trainval_data_ids(car_makers_list):
    mask = METADATA_CSV['make'].isin(car_makers_list) & METADATA_CSV['id'].isin(TRAIN_MASKS_CSV['id'])
    _ids = list(METADATA_CSV[mask]['id'])
    data_ids = list([None]*(len(_ids)*16))
    for i, _id in enumerate(_ids):
        for j in range(1, 17):
            data_ids[i*16 + j - 1] = (_id + "_{:02d}".format(j), "Train")
    return data_ids


def to_set(id_type_list):
    return set([(i[0], i[1]) for i in id_type_list])


def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type
    """
    check_dir = False
    if "Train" == image_type:
        ext = 'jpg'
        data_path = TRAIN_DATA
        suffix = ''
    elif "Train_mask" in image_type:
        ext = 'gif'
        data_path = TRAIN_MASKS_DATA
        suffix = '_mask'
    elif "Test" in image_type:
        ext = 'jpg'
        data_path = TEST_DATA
        suffix = ''
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    if check_dir and not os.path.exists(data_path):
        os.makedirs(data_path)

    return os.path.join(data_path, "{}{}.{}".format(image_id, suffix, ext))


# def get_metadata(image_id, image_type):
#     assert "Train" in image_type, "Can get only train caption"
#
#     return METADATA_CSV.loc[int(image_id), 'tags']


# def get_rle_mask(image_id, image_type, as_series=False, class_index=None, tag=None):
#     assert "Train" in image_type, "Can get only train labels"
#     assert not (class_index is not None and tag is not None), "Either class_index or either tag, not both"
#     if class_index is not None:
#         tags = equalized_data_classes[class_index]
#     elif tag is not None:
#         tags = [tag, ]
#     else:
#         tags = unique_tags
#
#     if "Generated" in image_type:
#         pass
#         _image_ids = image_id.split('_')
#         _image_type = image_type[len("Generated_"):]
#         _label_1 = get_label(_image_ids[0], _image_type, as_series=as_series)
#         _label_2 = get_label(_image_ids[1], _image_type, as_series=as_series)
#         _label = np.clip(_label_1 + _label_2, 0, 1)
#         if as_series:
#             return _label[tags]
#         else:
#             return _label[np.where(np.isin(np_unique_tags, tags))]
#     else:
#         if as_series:
#             return TRAIN_ENC_CSV.loc[int(image_id), tags]
#         return TRAIN_ENC_CSV.loc[int(image_id), tags].values.astype(np.uint8)


# def get_class_label_mask(class_index):
#     out = np.zeros(len(unique_tags), dtype=np.uint8)
#     for name in equalized_data_classes[class_index]:
#         out[unique_tags.index(name)] = 1
#     return out


def find_best_weights_file(weights_files, field_name='val_loss', best_min=True):

    if best_min:
        best_value = 1e5
        comp = lambda a, b: a > b
    else:
        best_value = -1e5
        comp = lambda a, b: a < b

    if '=' != field_name[-1]:
        field_name += '='

    best_weights_filename = ""
    for f in weights_files:
        index = f.find(field_name)
        index += len(field_name)
        assert index >= 0, "Field name '%s' is not found in '%s'" % (field_name, f)
        end = f.find('_', index)
        val = float(f[index:end])
        if comp(best_value, val):
            best_value = val
            best_weights_filename = f
    return best_weights_filename, best_value


def load_pretrained_model(model, by_name=False, **params):

    assert 'pretrained_model' in params, "pretrained_model is needed"
    assert 'save_prefix' in params, "save_prefix is needed"

    if params['pretrained_model'] == 'load_best':
        weights_files = []
        weights_files.extend(glob(os.path.join(OUTPUT_PATH, "weights", "%s*.h5" % params['save_prefix'])))
        weights_files.extend(glob(os.path.join(RESOURCES_PATH, "%s*.h5" % params['save_prefix'])))
        assert len(weights_files) > 0, "Failed to load weights"
        best_weights_filename, best_val_loss = find_best_weights_file(weights_files, field_name='val_loss')
        print("Load best loss weights: ", best_weights_filename, best_val_loss)
        model.load_weights(best_weights_filename, by_name=by_name)
    else:
        assert os.path.exists(params['pretrained_model']), "Not found pretrained model : %s" % params['pretrained_model']
        print("Load weights: ", params['pretrained_model'])
        model.load_weights(params['pretrained_model'], by_name=by_name)


class DataCache(object):
    """
    Queue storage of any data to avoid reloading
    """
    def __init__(self, n_samples):
        """
        :param n_samples: max number of data items to store in RAM
        """
        self.n_samples = n_samples
        self.cache = {}
        self.ids_queue = []

    def put(self, data_id, data):

        if 0 < self.n_samples < len(self.cache):
            key_to_remove = self.ids_queue.pop(0)
            self.cache.pop(key_to_remove)

        self.cache[data_id] = data
        if data_id in self.ids_queue:
            self.ids_queue.remove(data_id)
        self.ids_queue.append(data_id)

    def get(self, data_id):
        return self.cache[data_id]

    def remove(self, data_id):
        self.ids_queue.remove(data_id)
        self.cache.pop(data_id)

    def __contains__(self, key):
        return key in self.cache and key in self.ids_queue


