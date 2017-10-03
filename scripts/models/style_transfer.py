from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
import lasagne
from layers import RGBtoBGRLayer
import theano.tensor as T
import numpy as np
import pickle
from constants import *


MEAN_VALUES = np.array([104, 117, 123])
# Note: tweaked to use average pooling instead of maxpooling
def build_model(height, width, input_var):
    net = {}
    net['input_s'] = InputLayer((None, 3, height, width), input_var = input_var)
    net['conv1_1_s'] = ConvLayer(net['input_s'], 64, 3, pad=1, flip_filters=False)
    net['conv1_1_s'].params[net['conv1_1_s'].W].remove('trainable')
    net['conv1_1_s'].params[net['conv1_1_s'].b].remove('trainable')
    net['conv1_2_s'] = ConvLayer(net['conv1_1_s'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2_s'].params[net['conv1_2_s'].W].remove('trainable')
    net['conv1_2_s'].params[net['conv1_2_s'].b].remove('trainable')
    net['pool1_s'] = PoolLayer(net['conv1_2_s'], 2, mode='average_exc_pad')

    net['conv2_1_s'] = ConvLayer(net['pool1_s'], 128, 3, pad=1, flip_filters=False)
    net['conv2_1_s'].params[net['conv2_1_s'].W].remove('trainable')
    net['conv2_1_s'].params[net['conv2_1_s'].b].remove('trainable')
    net['conv2_2_s'] = ConvLayer(net['conv2_1_s'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2_s'].params[net['conv2_2_s'].W].remove('trainable')
    net['conv2_2_s'].params[net['conv2_2_s'].b].remove('trainable')
    net['pool2_s'] = PoolLayer(net['conv2_2_s'], 2, mode='average_exc_pad')
    
    net['conv3_1_s'] = ConvLayer(net['pool2_s'], 256, 3, pad=1, flip_filters=False)
    net['conv3_1_s'].params[net['conv3_1_s'].W].remove('trainable')
    net['conv3_1_s'].params[net['conv3_1_s'].b].remove('trainable')
    net['conv3_2_s'] = ConvLayer(net['conv3_1_s'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2_s'].params[net['conv3_2_s'].W].remove('trainable')
    net['conv3_2_s'].params[net['conv3_2_s'].b].remove('trainable')
    net['conv3_3_s'] = ConvLayer(net['conv3_2_s'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3_s'].params[net['conv3_3_s'].W].remove('trainable')
    net['conv3_3_s'].params[net['conv3_3_s'].b].remove('trainable')
    net['conv3_4_s'] = ConvLayer(net['conv3_3_s'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4_s'].params[net['conv3_4_s'].W].remove('trainable')
    net['conv3_4_s'].params[net['conv3_4_s'].b].remove('trainable')
    net['pool3_s'] = PoolLayer(net['conv3_4_s'], 2, mode='average_exc_pad')
    
    net['conv4_1_s'] = ConvLayer(net['pool3_s'], 512, 3, pad=1, flip_filters=False)
    net['conv4_1_s'].params[net['conv4_1_s'].W].remove('trainable')
    net['conv4_1_s'].params[net['conv4_1_s'].b].remove('trainable')
    net['conv4_2_s'] = ConvLayer(net['conv4_1_s'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2_s'].params[net['conv4_2_s'].W].remove('trainable')
    net['conv4_2_s'].params[net['conv4_2_s'].b].remove('trainable')
    net['conv4_3_s'] = ConvLayer(net['conv4_2_s'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3_s'].params[net['conv4_3_s'].W].remove('trainable')
    net['conv4_3_s'].params[net['conv4_3_s'].b].remove('trainable')
    net['conv4_4_s'] = ConvLayer(net['conv4_3_s'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4_s'].params[net['conv4_4_s'].W].remove('trainable')
    net['conv4_4_s'].params[net['conv4_4_s'].b].remove('trainable')
    net['pool4_s'] = PoolLayer(net['conv4_4_s'], 2, mode='average_exc_pad')
    '''
    net['conv5_1_s'] = ConvLayer(net['pool4_s'], 512, 3, pad=1, flip_filters=False)
    net['conv5_1_s'].params[net['conv5_1_s'].W].remove('trainable')
    net['conv5_1_s'].params[net['conv5_1_s'].b].remove('trainable')
    net['conv5_2_s'] = ConvLayer(net['conv5_1_s'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2_s'].params[net['conv5_2_s'].W].remove('trainable')
    net['conv5_2_s'].params[net['conv5_2_s'].b].remove('trainable')
    net['conv5_3_s'] = ConvLayer(net['conv5_2_s'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3_s'].params[net['conv5_3_s'].W].remove('trainable')
    net['conv5_3_s'].params[net['conv5_3_s'].b].remove('trainable')
    net['conv5_4_s'] = ConvLayer(net['conv5_3_s'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4_s'].params[net['conv5_4_s'].W].remove('trainable')
    net['conv5_4_s'].params[net['conv5_4_s'].b].remove('trainable')
    net['pool5_s'] = PoolLayer(net['conv5_4_s'], 2, mode='average_exc_pad')
    '''
    return net
    
def connect(net):
  #  net['input_s'] = InputLayer((1, 3, IMAGE_W, IMAGE_W))
    net['bgr_s'] = RGBtoBGRLayer(net['output_encoder_scaled'], bgr_mean=MEAN_VALUES, data_format='bc01')
    print "bgr_s: {}".format(net['bgr_s'].output_shape[1:])
    
    net['conv1_1_s'] = ConvLayer(net['bgr_s'], 64, 3, pad=1, flip_filters=False)
    net['conv1_1_s'].params[net['conv1_1_s'].W].remove('trainable')
    net['conv1_1_s'].params[net['conv1_1_s'].b].remove('trainable')
    net['conv1_2_s'] = ConvLayer(net['conv1_1_s'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2_s'].params[net['conv1_2_s'].W].remove('trainable')
    net['conv1_2_s'].params[net['conv1_2_s'].b].remove('trainable')
    net['pool1_s'] = PoolLayer(net['conv1_2_s'], 2, mode='average_exc_pad')
    
    net['conv2_1_s'] = ConvLayer(net['pool1_s'], 128, 3, pad=1, flip_filters=False)
    net['conv2_1_s'].params[net['conv2_1_s'].W].remove('trainable')
    net['conv2_1_s'].params[net['conv2_1_s'].b].remove('trainable')
    net['conv2_2_s'] = ConvLayer(net['conv2_1_s'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2_s'].params[net['conv2_2_s'].W].remove('trainable')
    net['conv2_2_s'].params[net['conv2_2_s'].b].remove('trainable')
    net['pool2_s'] = PoolLayer(net['conv2_2_s'], 2, mode='average_exc_pad')
    
    net['conv3_1_s'] = ConvLayer(net['pool2_s'], 256, 3, pad=1, flip_filters=False)
    net['conv3_1_s'].params[net['conv3_1_s'].W].remove('trainable')
    net['conv3_1_s'].params[net['conv3_1_s'].b].remove('trainable')
    net['conv3_2_s'] = ConvLayer(net['conv3_1_s'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2_s'].params[net['conv3_2_s'].W].remove('trainable')
    net['conv3_2_s'].params[net['conv3_2_s'].b].remove('trainable')
    net['conv3_3_s'] = ConvLayer(net['conv3_2_s'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3_s'].params[net['conv3_3_s'].W].remove('trainable')
    net['conv3_3_s'].params[net['conv3_3_s'].b].remove('trainable')
    net['conv3_4_s'] = ConvLayer(net['conv3_3_s'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4_s'].params[net['conv3_4_s'].W].remove('trainable')
    net['conv3_4_s'].params[net['conv3_4_s'].b].remove('trainable')
    net['pool3_s'] = PoolLayer(net['conv3_4_s'], 2, mode='average_exc_pad')
    
    net['conv4_1_s'] = ConvLayer(net['pool3_s'], 512, 3, pad=1, flip_filters=False)
    net['conv4_1_s'].params[net['conv4_1_s'].W].remove('trainable')
    net['conv4_1_s'].params[net['conv4_1_s'].b].remove('trainable')
    net['conv4_2_s'] = ConvLayer(net['conv4_1_s'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2_s'].params[net['conv4_2_s'].W].remove('trainable')
    net['conv4_2_s'].params[net['conv4_2_s'].b].remove('trainable')
    net['conv4_3_s'] = ConvLayer(net['conv4_2_s'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3_s'].params[net['conv4_3_s'].W].remove('trainable')
    net['conv4_3_s'].params[net['conv4_3_s'].b].remove('trainable')
    net['conv4_4_s'] = ConvLayer(net['conv4_3_s'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4_s'].params[net['conv4_4_s'].W].remove('trainable')
    net['conv4_4_s'].params[net['conv4_4_s'].b].remove('trainable')
    net['pool4_s'] = PoolLayer(net['conv4_4_s'], 2, mode='average_exc_pad')
    '''  
    net['conv5_1_s'] = ConvLayer(net['pool4_s'], 512, 3, pad=1, flip_filters=False)
    net['conv5_1_s'].params[net['conv5_1_s'].W].remove('trainable')
    net['conv5_1_s'].params[net['conv5_1_s'].b].remove('trainable')
    net['conv5_2_s'] = ConvLayer(net['conv5_1_s'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2_s'].params[net['conv5_2_s'].W].remove('trainable')
    net['conv5_2_s'].params[net['conv5_2_s'].b].remove('trainable')
    net['conv5_3_s'] = ConvLayer(net['conv5_2_s'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3_s'].params[net['conv5_3_s'].W].remove('trainable')
    net['conv5_3_s'].params[net['conv5_3_s'].b].remove('trainable')
    net['conv5_4_s'] = ConvLayer(net['conv5_3_s'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4_s'].params[net['conv5_4_s'].W].remove('trainable')
    net['conv5_4_s'].params[net['conv5_4_s'].b].remove('trainable')
    net['pool5_s'] = PoolLayer(net['conv5_4_s'], 2, mode='average_exc_pad')
    '''
    return net

def load_weights(net):
    print ("Loading Style net weights ...")
    values = pickle.load(open('vgg19_normalized.pkl'))['param values']
    layers = ['conv1_1_s', 'conv1_2_s', 'conv2_1_s', 'conv2_2_s',
              'conv3_1_s', 'conv3_2_s', 'conv3_3_s', 'conv3_4_s', 
              'conv4_1_s', 'conv4_2_s', 'conv4_3_s', 'conv4_4_s']

              #'conv5_1_s', 'conv5_2_s', 'conv5_3_s', 'conv5_4_s']
    layers = [ net[k].get_params() for k in layers]

    for i in range(len(layers)):
        if len(layers[i]) == 2:
            layers[i][0].set_value(values[i*2])
            layers[i][1].set_value(values[i*2+1])
    #lasagne.layers.set_all_param_values(layers.values(), values)


def prep_image(im):
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    #h, w, _ = im.shape
    #if h < w:
    #    im = skimage.transform.resize(im, (IMAGE_W, w*IMAGE_W/h), preserve_range=True)
    #else:
    #    im = skimage.transform.resize(im, (h*IMAGE_W/w, IMAGE_W), preserve_range=True)

    # Central crop
    #h, w, _ = im.shape
    #im = im[h//2-IMAGE_W//2:h//2+IMAGE_W//2, w//2-IMAGE_W//2:w//2+IMAGE_W//2]
    
    #rawim = np.copy(im).astype('uint8')
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert RGB to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUES.reshape(3,1,1)
    return im

def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g
def content_loss(P, X):
    distance = lasagne.objectives.squared_error(P, X)    
    loss = 1./2 * distance.sum()
    return loss

def style_loss(a, x):
                
    A = gram_matrix(a)
    G = gram_matrix(x)
                        
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    distance = lasagne.objectives.squared_error(G, A)
    loss = 1./(4 * N**2 * M**2) * distance.sum()
    return loss

