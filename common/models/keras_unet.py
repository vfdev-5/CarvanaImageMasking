from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Deconv2D, LeakyReLU, Concatenate, Activation, Input
from keras.models import Model


def get_unet(**kwargs):
    """
     U-Net model with Conv->BN->LeakyReLU
    """
    assert 'input_shape' in kwargs, "Needed input_shape, e.g (256, 256, 3)"
    assert 'n_classes' in kwargs, "Needed n_classes, e.g. 3"
    input_shape = kwargs['input_shape']
    n_classes = kwargs['n_classes']
    depth = 5 if 'depth' not in kwargs else kwargs['depth']
    n_filters = 32 if 'n_filters' not in kwargs else kwargs['n_filters']

    inputs = Input(name="data", shape=input_shape)
    list_encoder, list_nb_filters = encoder(inputs, n_filters, depth=depth)
    x = decoder(list_encoder, list_nb_filters)

    x = Conv2D(n_classes,
               kernel_size=(1, 1),
               name="output_conv")(x)
    outputs = Activation("sigmoid", name="output_proba")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def base_conv(x, n_filters, s_id=""):
    x = Conv2D(n_filters,
               kernel_size=(3, 3),
               padding='same',
               use_bias=False,
               name="%s_conv" % s_id)(x)
    x = BatchNormalization(name="%s_bn" % s_id)(x)
    return LeakyReLU(alpha=0.25, name="%s_act" % s_id)(x)


def encoder(inputs, n_filters, depth=5):
    list_encoder = []
    list_nb_filters = []
    temp_layers = [inputs]
    for i in range(depth):
        nf = n_filters * 2 ** i
        x = base_conv(temp_layers[-1], nf, s_id="encoder_%ia" % i)
        x = base_conv(x, nf, s_id="encoder_%ib" % i)
        list_encoder.append(x)
        list_nb_filters.append(nf)
        x = MaxPool2D(pool_size=(2, 2), name="encoder_%i_pool" % i)(x)
        temp_layers.append(x)
    i = depth
    nf = n_filters * 2 ** i
    x = base_conv(temp_layers[-1], nf, s_id="encoder_%ia" % i)
    x = base_conv(x, nf, s_id="encoder_%ib" % i)
    list_encoder.append(x)
    return list_encoder, list_nb_filters


def decoder(list_encoder, list_nb_filters):
    l = len(list_encoder)
    temp_layers = [list_encoder[l - 1]]
    for i in range(l - 1):
        x = Deconv2D(list_nb_filters[l - i - 2],
                     kernel_size=(2, 2),
                     padding='same',
                     strides=(2, 2),
                     name="decoder_%i_deconv" % i)(temp_layers[-1])
        x = Concatenate(name="decoder_%i_concat" % i)([x, list_encoder[l - i - 2]])
        x = base_conv(x, list_nb_filters[l - i - 2], s_id="decoder_%ia" % i)
        x = base_conv(x, list_nb_filters[l - i - 2], s_id="decoder_%ib" % i)
        temp_layers.append(x)
    return temp_layers[-1]
