import tensorflow_compression as tfc
from keras.layers import Input, PReLU, Activation, GlobalAveragePooling2D, Dense, Concatenate, Conv2D, Multiply, MaxPooling2D, Flatten, UpSampling2D, Dropout
from tensorflow import keras

def GFR_Encoder_Module(inputs, name_prefix, num_filter, kernel_size, stride, activation=None):
    conv = tfc.SignalConv2D(num_filter, kernel_size, corr=True, strides_down=stride, padding="same_zeros",
                            use_bias=True, activation=tfc.GDN(), name=name_prefix + '_conv')(inputs)
    if activation == 'prelu':
        conv = PReLU(shared_axes=[1,2], name=name_prefix + '_prelu')(conv)
    return conv

def Modified_Basic_Encoder(inputs, tcn):
    en1 = GFR_Encoder_Module(inputs, 'en1', 64, (3, 3), 1, 'prelu')
    en2 = GFR_Encoder_Module(en1, 'en2', 64, (3, 3), 1, 'prelu')
    x = MaxPooling2D((2, 2), strides=(2, 2))(en2)

    en3 = GFR_Encoder_Module(x, 'en3', 128, (3, 3), 1, 'prelu')
    en4 = GFR_Encoder_Module(en3, 'en4', 128, (3, 3), 1, 'prelu')
    x = MaxPooling2D((2, 2), strides=(2, 2))(en4)

    en5 = GFR_Encoder_Module(x, 'en5', 256, (3, 3), 1, 'prelu')
    en6 = GFR_Encoder_Module(en5, 'en6', 256, (3, 3), 1, 'prelu')
    en7 = GFR_Encoder_Module(en6, 'en7', 256, (3, 3), 1, 'prelu')
    x = MaxPooling2D((2, 2), strides=(2, 2))(en7)

    en8 = GFR_Encoder_Module(x, 'en8', tcn, (3, 3), 1,)
    return en8

def GFR_Decoder_Module(inputs, name_prefix, num_filter, kernel_size, stride, activation=None):
    conv = tfc.SignalConv2D(num_filter, kernel_size, corr=False, strides_up=stride, padding="same_zeros", use_bias=True,
                            activation=tfc.GDN(inverse=True), name=name_prefix + '_conv')(inputs)
    if activation == 'prelu':
        conv = PReLU(shared_axes=[1,2], name=name_prefix + '_prelu')(conv)
    elif activation == 'sigmoid':
        conv = Activation('sigmoid', name=name_prefix + '_sigmoid')(conv)
    return conv

def Modified_Basic_Decoder(channel_output):
    de1 = GFR_Decoder_Module(channel_output, 'de1', 64, (3, 3), 1, 'prelu')
    de2 = GFR_Decoder_Module(de1, 'de2', 64, (3, 3), 1, 'prelu')
    x = MaxPooling2D((2, 2), strides = (2, 2))(de2)

    de3 = GFR_Decoder_Module(x, 'de3', 128, (3, 3), 1, 'prelu')
    de4 = GFR_Decoder_Module(de3, 'de4', 128, (3, 3), 1, 'prelu')
    x = MaxPooling2D((2, 2), strides = (2, 2))(de4)

    de5 = GFR_Decoder_Module(x, 'de5', 256, (3, 3), 1, 'prelu')
    de6 = GFR_Decoder_Module(de5, 'de6', 256, (3, 3), 1, 'prelu')
    de7 = GFR_Decoder_Module(de6, 'de7', 256, (3, 3), 1, 'prelu')
    if de7.shape[1] > 4 and de7.shape[2] > 4:
        x = MaxPooling2D((2, 2), strides=(2, 2))(de7)

    x = Flatten()(x)

    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    prediction =  Dense(10, activation='softmax')(x)

    return prediction