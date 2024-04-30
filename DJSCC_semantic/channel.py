import tensorflow as tf
import numpy as np
import keras.layers as layers

class Channel(layers.Layer):

    def __init__(self, channel_type, name="channel", **kwargs):
        super(Channel, self).__init__(name=name, **kwargs)
        self.channel_type = channel_type

    def call(self, features, snr_db=None, h_real=None, h_imag=None):
        shape = tf.shape(features)
        features_fla = layers.Flatten()(features)

        # transform to complex
        z_dim = tf.shape(features_fla)[1] // 2
        z_input = tf.complex(features_fla[:, :z_dim], features_fla[:, z_dim:])

        # normalize power
        norm_index = tf.reduce_sum(tf.math.real(z_input * tf.math.conj(z_input)), axis=1, keepdims=True)
        z_input_nor = z_input * tf.complex(tf.sqrt(tf.cast(z_dim, dtype=tf.float32) / norm_index), 0.0)

        # add channel noise
        if self.channel_type == 'awgn':
            if snr_db is None:
                raise Exception("NO SNR CONDITION PARAMETER")
            z_output = awgn(z_input_nor, snr_db)
        elif self.channel_type == 'slow_fading':
            if snr_db is None or h_real is None or h_imag is None:
                raise Exception("NO SNR AND H CONDITION PARAMETERS")
            z_output = slow_fading(z_input_nor, snr_db, h_real, h_imag)
        else:
            raise Exception("NO CHANNEL CONDITION PARAMETER")
        
        # transform to normal
        z_output = tf.concat([tf.math.real(z_output), tf.math.imag(z_output)], 1)
        z_output = tf.reshape(z_output, shape)
        return z_output


def awgn(x, snr_db):
    noise_nor = tf.sqrt(10 ** (-snr_db / 10))
    noise_stdc = tf.complex(noise_nor, 0.)
    awgn = tf.complex(tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)), tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),)
    
    return x + noise_stdc * awgn


def slow_fading(x, snr_db, h_real, h_imag):
    noise_nor = tf.sqrt(10 ** (-snr_db / 10))
    noise_std = tf.complex(noise_nor, 0.)
    h = tf.complex(h_real, h_imag)
    h = tf.reshape(h, (tf.shape(h)[0], 1))
    awgn = tf.complex(tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)), tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),)
    
    return h * x + noise_std * awgn