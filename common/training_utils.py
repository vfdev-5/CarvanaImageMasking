
from __future__ import absolute_import

import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, Callback

from .data_utils import GENERATED_DATA, OUTPUT_PATH, to_set


def exp_decay(epoch, lr=1e-3, a=0.925, init_epoch=0):
    return lr * np.exp(-(1.0 - a) * (epoch + init_epoch))


def step_decay(epoch, lr=1e-3, base=2.0, period=50, init_epoch=0):
    return lr * base ** (-np.floor((epoch + init_epoch) * 1.0 / period))


def write_info(filename, **kwargs):
    with open(filename, 'w') as f:
        for k in kwargs:
            f.write("{}: {}\n".format(k, kwargs[k]))


class EpochValidationCallback(Callback):

    def __init__(self, val_id_type_list, **params):
        super(EpochValidationCallback, self).__init__()
        self.val_id_type_list = val_id_type_list
        self.ev_params = dict(params)
        self.ev_params['verbose'] = 0
        if 'EpochValidationCallback_rate' not in self.ev_params:
            self.ev_params['EpochValidationCallback_rate'] = 3
        assert 'seed' in self.ev_params, "Need seed, params: {}".format(self.ev_params)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.ev_params['EpochValidationCallback_rate'] > 0:
            return
        # f2, mae = classification_validate(self.model, self.val_id_type_list, **self.ev_params)
        # print("\nEpoch validation: f2 = %f, mae=%f \n" % (f2, mae))


def cv_training(trainval_ids, n_folds=5, val_fold_indices=(), **kwargs):
    """
    :param trainval_ids:
    :param n_folds:
    :param val_fold_indices: (optional) a list of validation folds indices to run training on specific folds.
    If not specified training is run on all folds.
    :param kwargs:
    :return:
    """

    val_fold_index = 0
    assert isinstance(trainval_ids, list) or isinstance(trainval_ids, tuple), "trainval_ids should be a list or tuple"

    kf = KFold(n_splits=n_folds)
    trainval_ids = np.array(trainval_ids)
    for train_index, test_index in kf.split(trainval_ids):
        train_ids, val_ids = trainval_ids[train_index], trainval_ids[test_index]
        if len(val_fold_indices) > 0:
            if val_fold_index not in val_fold_indices:
                val_fold_index += 1
                continue
        assert len(to_set(train_ids) & to_set(val_ids)) == 0, "Train and validation data ids have common ids"
        val_fold_index += 1
        print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)
        training(train_ids, val_ids, **kwargs)


# def _check_training_kwargs(**kwargs):
#     # Check kwargs, required keys:
#     # - network
#     assert 'network' in kwargs, "CNN model is required in kwargs"

def training(train_ids, val_ids, params, custom_objects):
    """
    :param train_ids: list or tuple of data identifiers for training dataset
    :param val_ids: list or tuple of data identifiers for validation dataset
    :param params: dictionary with following configuration blocks:
        'data_provider':

    :param custom_objects: dictionary of objects used in params
    :return:
    """
    # _check_training_kwargs(**kwargs)

    # Setup and compile CNN model


def segmentation_train(model,
                       train_id_type_list,
                       val_id_type_list,
                       **kwargs):

    params = dict(kwargs)
    assert 'batch_size' in params, "Need batch_size"
    assert 'save_prefix' in params, "Need save_prefix"
    assert 'nb_epochs' in params, "Need nb_epochs"
    assert 'seed' in params, "Need seed"
    assert 'normalize_data' in params, "Need normalize_data"

    samples_per_epoch = len(train_id_type_list) if 'samples_per_epoch' not in params else params['samples_per_epoch']
    nb_val_samples = len(val_id_type_list) if 'nb_val_samples' not in params else params['nb_val_samples']
    lr_decay_f = None if 'lr_decay_f' not in params else params['lr_decay_f']

    save_prefix = params['save_prefix']
    batch_size = params['batch_size']
    nb_epochs = params['nb_epochs']
    seed = params['seed']
    normalize_data = params['normalize_data']
    if normalize_data:
        assert 'normalization' in params, "Need normalization"
        normalization = params['normalization']
    else:
        normalization = None

    samples_per_epoch = (samples_per_epoch // batch_size + 1) * batch_size
    nb_val_samples = (nb_val_samples // batch_size + 1) * batch_size

    output_path = params['output_path'] if 'output_path' in params else GENERATED_DATA
    weights_path = os.path.join(output_path, "weights")
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    weights_filename = os.path.join(weights_path,
                                    save_prefix + "_{epoch:02d}_val_loss={val_loss:.4f}")

    metrics_names = list(model.metrics_names)
    metrics_names.remove('loss')
    for mname in metrics_names:
        weights_filename += "_val_%s={val_%s:.4f}" % (mname, mname)
    weights_filename += ".h5"

    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss',
                                       save_best_only=False, save_weights_only=False)
    now = datetime.now()
    info_filename = os.path.join(weights_path,
                                 'training_%s_%s.info' % (save_prefix, str(now.strftime("%Y-%m-%d-%H-%M"))))

    write_info(info_filename, **params)

    csv_logger = CSVLogger(os.path.join(weights_path,
                                        'training_%s_%s.log' % (save_prefix, str(now.strftime("%Y-%m-%d-%H-%M")))))

    epoch_validation = EpochValidationCallback(val_id_type_list, **params)

    callbacks = [model_checkpoint, csv_logger, epoch_validation]
    if lr_decay_f is not None:
        assert 'lr_kwargs' in params and \
               isinstance(params['lr_kwargs'], dict), "Need lr_kwargs"
        _lr_decay_f = lambda e: lr_decay_f(epoch=e, **params['lr_kwargs'])
        lrate = LearningRateScheduler(_lr_decay_f)
        callbacks.append(lrate)
    if 'on_plateau' in params and params['on_plateau']:
        if 'on_plateau_kwargs' in params and \
                isinstance(params['lr_kwargs'], dict):
            onplateau = ReduceLROnPlateau(**params['on_plateau_kwargs'])
        else:
            onplateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
        callbacks.append(onplateau)

    print("\n-- Training parameters: %i, %i, %i, %i" % (batch_size, nb_epochs, samples_per_epoch, nb_val_samples))
    print("\n-- Fit model")

    class_weight = params.get('class_weight')
    verbose = 1 if 'verbose' not in params else params['verbose']

    if 'train_seq' in params:
        assert callable(params['train_seq']), "params['train_seq'] should be callable"
    train_seq = get_imgaug_seq(seed) if 'train_seq' not in params else params['train_seq'](seed)

    if 'val_seq' in params:
        assert callable(params['val_seq']), "params['val_seq'] should be callable"
    val_seq = get_imgaug_seq(seed) if 'val_seq' not in params else params['val_seq'](seed)

    try:
        train_gen, train_flow = get_gen_flow(id_type_list=train_id_type_list,
                                             imgaug_seq=train_seq,
                                             **params)

        if normalize_data and normalization == '':
            params['normalization'] = 'from_save_prefix'

        val_gen, val_flow = get_gen_flow(id_type_list=val_id_type_list,
                                         imgaug_seq=val_seq,
                                         **params)

        np.random.seed(seed)
        # New or old Keras API
        history = model.fit_generator(generator=train_flow,
                                      steps_per_epoch=(samples_per_epoch // batch_size),
                                      epochs=nb_epochs,
                                      validation_data=val_flow,
                                      validation_steps=(nb_val_samples // batch_size),
                                      callbacks=callbacks,
                                      class_weight=class_weight,
                                      verbose=verbose)
        return history
    except KeyboardInterrupt:
        pass


# def segmentation_validate(model,
#                           val_id_type_list,
#                           **kwargs):
#
#     params = dict(kwargs)
#     assert 'seed' in params, "Need seed, params = {}".format(params)
#     assert 'normalize_data' in params, "Need normalize_data"
#     verbose = 1 if 'verbose' not in params else params['verbose']
#     save_predictions = False if 'save_predictions' not in params else params['save_predictions']
#     save_predictions_id = '' if 'save_predictions_id' not in params else params['save_predictions_id']
#     n_classes = len(unique_tags) if 'n_classes' not in params else params['n_classes']
#
#     normalize_data = params['normalize_data']
#     if normalize_data:
#         assert 'normalization' in params, "Need normalization"
#         normalization = params['normalization']
#     else:
#         normalization = None
#
#     if normalize_data and normalization == '':
#         params['normalization'] = 'from_save_prefix'
#
#     val_seq = get_val_imgaug_seq(params['seed'])
#     val_gen, val_flow = get_gen_flow(id_type_list=val_id_type_list,
#                                      imgaug_seq=val_seq,
#                                      test_mode=True, **params)
#
#     y_true_total = np.zeros((len(val_id_type_list), n_classes))
#     y_pred_total = np.zeros_like(y_true_total)
#     info_total = np.empty((y_true_total.shape[0], ), dtype=np.object)
#     counter = 0
#     for x, y_true, info in val_flow:
#         if verbose > 0:
#             print("-- %i / %i" % (counter, len(val_id_type_list)), info)
#         s = y_true.shape[0]
#         start = counter * s
#         end = min((counter + 1) * s, len(val_id_type_list))
#         y_true_total[start:end, :] = y_true
#         info_total[start:end] = ['train_' + i[0] for i in info]
#
#         y_pred = model.predict(x)
#         y_pred_total[start:end, :] = y_pred
#
#         counter += 1
#
#     if save_predictions:
#         df = pd.DataFrame(columns=('image_name',) + tuple(unique_tags))
#         df['image_name'] = info_total
#         df[unique_tags] = y_pred_total
#         df.to_csv(os.path.join(OUTPUT_PATH, 'val_predictions_' + save_predictions_id + '.csv'), index=False)
#         if verbose > 0:
#             print("Saved predictions with id: %s" % save_predictions_id)
#
#     y_pred_total2 = pred_threshold(y_pred_total)
#     total_f2 = score(y_true_total, y_pred_total2)
#     total_mae = mean_absolute_error(y_true_total, y_pred_total2)
#
#     if verbose > 0:
#         print("Total f2, mae : ", total_f2, total_mae)
#     return total_f2, total_mae
