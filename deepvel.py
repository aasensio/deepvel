import numpy as np
import platform
import os
from astropy.io import fits
import time
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'vena'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.layers import Input, Convolution2D, merge, Activation, BatchNormalization
from keras.models import Model


class deepvel(object):

    def __init__(self, observations, output, border=0):
        """

        Parameters
        ----------
        observations : array
            Array of size (n_times, nx, ny) with the n_times consecutive images of size nx x ny
        output : string
            Filename were the output is saved
        border : int (optional)
            Portion of the borders to be removed during computations. This is useful if images are
            apodized
        """

# Only allocate needed memory with Tensorflow
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        session = tf.Session(config=config)
        ktf.set_session(session)

        self.border = border
        n_timesteps, nx, ny = observations.shape

        self.n_frames = n_timesteps - 1

        self.nx = nx - 2*self.border
        self.ny = ny - 2*self.border
        
        self.n_times = 2
        self.n_filters = 64
        self.batch_size = 1
        self.n_conv_layers = 20        
        self.observations = observations
        self.output = output

        print("Images without border are of size: {0}x{1}".format(self.nx, self.ny))
        print("Number of predictions to be made: {0}".format(self.n_frames))

    def residual(self, inputs):
        x = Convolution2D(self.n_filters, 3, 3, border_mode='same', init='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(self.n_filters, 3, 3, border_mode='same', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = merge([x, inputs], 'sum')

        return x
            
    def define_network(self):
        print("Setting up network...")

        inputs = Input(shape=(self.nx, self.ny, self.n_times))
        conv = Convolution2D(self.n_filters, 3, 3, activation='relu', border_mode='same', init='he_normal')(inputs)

        x = self.residual(conv)
        for i in range(self.n_conv_layers):
            x = self.residual(x)

        x = Convolution2D(self.n_filters, 3, 3, border_mode='same', init='he_normal')(x)
        x = BatchNormalization()(x)
        x = merge([x, conv], 'sum')

        final = Convolution2D(6, 1, 1, activation='linear', border_mode='same', init='he_normal')(x)

        self.model = Model(input=inputs, output=final)
        self.model.load_weights('network/deepvel_weights.hdf5')
                        
    def validation_generator(self):        
        self.median_i = np.median(self.observations[:,self.border:-self.border,self.border:-self.border])

        input_validation = np.zeros((self.batch_size,self.nx,self.ny,2), dtype='float32')

        while 1:
            for i in range(self.n_frames):

                input_validation[:,:,:,0] = self.observations[i*self.batch_size:(i+1)*self.batch_size,self.border:-self.border,self.border:-self.border] / self.median_i
                input_validation[:,:,:,1] = self.observations[i*self.batch_size+1:(i+1)*self.batch_size+1,self.border:-self.border,self.border:-self.border] / self.median_i

                yield input_validation

        f.close()

    def predict(self):
        print("Predicting velocities with DeepVel...")

        tmp = np.load('network/normalization.npz')
        _, _, min_v, max_v = tmp['arr_0'], tmp['arr_1'], tmp['arr_2'], tmp['arr_3']

        start = time.time()
        out = self.model.predict_generator(self.validation_generator(), self.n_frames, max_q_size=1)
        end = time.time()

        print("Prediction took {0} seconds...".format(end-start))
        
        for i in range(6):
            out[:,:,:,i] = out[:,:,:,i] * (max_v[i] - min_v[i]) + min_v[i]

# This factor 10 comes from a change of units carried out during the simulations. Here we transform it back
        out *= 10

        hdu = fits.PrimaryHDU(out)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(self.output, overwrite=True)
        
    
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='DeepVel prediction')
    parser.add_argument('-o','--out', help='Output file')
    parser.add_argument('-i','--in', help='Input file')
    parser.add_argument('-b','--border', help='Border size in pixels', default=0)
    parsed = vars(parser.parse_args())

# Open file with observations and read them. We use FITS in our case
    f = fits.open(parsed['in'])
    imgs = f[0].data   
    
    out = deepvel(imgs, parsed['out'], border=int(parsed['border']))
    out.define_network()
    out.predict()