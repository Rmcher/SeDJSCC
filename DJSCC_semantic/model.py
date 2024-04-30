import tensorflow_compression as tfc
from keras.layers import Input, PReLU, Activation, GlobalAveragePooling2D, Dense, Concatenate, Conv2D, Multiply, MaxPooling2D, Flatten, UpSampling2D, Dropout

def Semantic_Encoder_classi(inputs, cn):
    x = tfc.SignalConv2D(64, (3, 3), corr=True, strides_down=1, padding="same_zeros", use_bias=True, activation=tfc.GDN())(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(64, (3, 3), corr=True, strides_down=1, padding="same_zeros", use_bias=True, activation=tfc.GDN())(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = tfc.SignalConv2D(128, (3, 3), corr=True, strides_down=1, padding="same_zeros", use_bias=True, activation=tfc.GDN())(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(128, (3, 3), corr=True, strides_down=1, padding="same_zeros", use_bias=True, activation=tfc.GDN())(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = tfc.SignalConv2D(256, (3, 3), corr=True, strides_down=1, padding="same_zeros", use_bias=True, activation=tfc.GDN())(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (3, 3), corr=True, strides_down=1, padding="same_zeros", use_bias=True, activation=tfc.GDN())(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (3, 3), corr=True, strides_down=1, padding="same_zeros", use_bias=True, activation=tfc.GDN())(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = tfc.SignalConv2D(cn, (3, 3), corr=True, strides_down=1, padding="same_zeros", use_bias=True, activation=None)(x)  # 最后一层可能不需要激活函数
    return x

def Semantic_Decoder_classi(inputs):
    # 使用 tfc 的 SignalConv2D 和 GDN 逆激活层
    x = tfc.SignalConv2D(64, (3, 3), corr=False, strides_up=1, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(64, (3, 3), corr=False, strides_up=1, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = tfc.SignalConv2D(128, (3, 3), corr=False, strides_up=1, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(128, (3, 3), corr=False, strides_up=1, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = tfc.SignalConv2D(256, (3, 3), corr=False, strides_up=1, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (3, 3), corr=False, strides_up=1, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = tfc.SignalConv2D(256, (3, 3), corr=False, strides_up=1, padding="same_zeros", use_bias=True, activation=tfc.GDN(inverse=True))(x)
    x = PReLU(shared_axes=[1, 2])(x)
    if x.shape[1] > 4 and x.shape[2] > 4:
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    return x
