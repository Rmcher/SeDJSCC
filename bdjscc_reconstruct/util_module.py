import tensorflow_compression as tfc
from keras.layers import PReLU, Activation, GlobalAveragePooling2D, Dense, Concatenate, Conv2D, Multiply


def GFR_Encoder_Module(inputs, name_prefix, num_filter, kernel_size, stride, activation=None):
    conv = tfc.SignalConv2D(num_filter, kernel_size, corr=True, strides_down=stride, padding="same_zeros",
                            use_bias=True, activation=tfc.GDN(), name=name_prefix + '_conv')(inputs)
    if activation == 'prelu':
        conv = PReLU(shared_axes=[1,2], name=name_prefix + '_prelu')(conv)
    return conv


def Basic_Encoder(inputs, tcn):
    en1 = GFR_Encoder_Module(inputs, 'en1', 256, (9, 9), 2, 'prelu')
    en2 = GFR_Encoder_Module(en1, 'en2', 256, (5, 5), 2, 'prelu')
    en3 = GFR_Encoder_Module(en2, 'en3', 256, (5, 5), 1, 'prelu')
    en4 = GFR_Encoder_Module(en3, 'en4', 256, (5, 5), 1, 'prelu')
    en5 = GFR_Encoder_Module(en4, 'en5', tcn, (5, 5), 1)
    return en5


def GFR_Decoder_Module(inputs, name_prefix, num_filter, kernel_size, stride, activation=None):
    conv = tfc.SignalConv2D(num_filter, kernel_size, corr=False, strides_up=stride, padding="same_zeros", use_bias=True,
                            activation=tfc.GDN(inverse=True), name=name_prefix + '_conv')(inputs)
    if activation == 'prelu':
        conv = PReLU(shared_axes=[1,2], name=name_prefix + '_prelu')(conv)
    elif activation == 'sigmoid':
        conv = Activation('sigmoid', name=name_prefix + '_sigmoid')(conv)
    return conv


def Basic_Decoder(inputs):
    de1 = GFR_Decoder_Module(inputs, 'de1', 256, (5, 5), 1, 'prelu')
    de2 = GFR_Decoder_Module(de1, 'de2', 256, (5, 5), 1, 'prelu')
    de3 = GFR_Decoder_Module(de2, 'de3', 256, (5, 5), 1, 'prelu')
    de4 = GFR_Decoder_Module(de3, 'de4', 256, (5, 5), 2, 'prelu')
    de5 = GFR_Decoder_Module(de4, 'de5', 3, (9, 9), 2, 'sigmoid')
    return de5