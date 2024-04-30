import tensorflow_compression as tfc
from keras.layers import PReLU

def Semantic_Encoder(inputs, cn):
    x = tfc.SignalConv2D(256, (9, 9), corr=True, strides_down=2, padding="same_zeros",
                         use_bias=True, activation=tfc.GDN())(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (5, 5), corr=True, strides_down=2, padding="same_zeros",
                         use_bias=True, activation=tfc.GDN())(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (5, 5), corr=True, strides_down=1, padding="same_zeros",
                         use_bias=True, activation=tfc.GDN())(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (5, 5), corr=True, strides_down=1, padding="same_zeros",
                         use_bias=True, activation=tfc.GDN())(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(cn, (5, 5), corr=True, strides_down=1, padding="same_zeros",
                         use_bias=True, activation=None)(x)
    return x

def Semantic_Decoder(inputs):
    x = tfc.SignalConv2D(256, (5, 5), corr=False, strides_up=1, padding="same_zeros",
                         use_bias=True, activation=tfc.GDN(inverse=True))(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (5, 5), corr=False, strides_up=1, padding="same_zeros",
                         use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (5, 5), corr=False, strides_up=1, padding="same_zeros",
                         use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (5, 5), corr=False, strides_up=2, padding="same_zeros",
                         use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(3, (9, 9), corr=False, strides_up=2, padding="same_zeros",
                         use_bias=True, activation='sigmoid')(x)
    return x
