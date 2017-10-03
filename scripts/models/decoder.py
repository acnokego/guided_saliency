from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Upscale2DLayer
from lasagne.nonlinearities import sigmoid
import lasagne
import cPickle
import vgg16
from constants import PATH_TO_VGG16_WEIGHTS
#from utils import *


def set_pretrained_weights(net, path_to_model_weights=PATH_TO_VGG16_WEIGHTS):
    # Set out weights
    vgg16 = cPickle.load(open(path_to_model_weights))
    print ("Loading vgg16 weights ...")
    num_elements_to_set = 26  # Number of W and b elements for the first convolutional layers
    '''
    layers = ['conv1_1_de', 'conv1_2_de', 'conv2_1_de', 'conv2_2_de', 
               'conv3_1_de', 'conv3_2_de', 'conv3_3_de', 
                'conv4_1_de', 'conv4_2_de', 'conv4_3_de', 
                 'conv5_1_de', 'conv5_2_de','conv5_3_de']
    layers = [ net[k].get_params() for k in layers]

    for i in range(len(layers)):
        if len(layers[i]) == 2:
            layers[i][0].set_value(vgg16['param values'][i*2])
            layers[i][1].set_value(vgg16['param values'][i*2+1])
    '''
    lasagne.layers.set_all_param_values(net['conv5_3_de'], vgg16['param values'][:num_elements_to_set])


def build_encoder(net, input_height, input_width):
    encoder = vgg16.build(None, input_height, input_width,connect=False)
    #set_pretrained_weights(encoder)
    return encoder


def build_decoder(net):
    net['uconv5_3_de']= ConvLayer(net['conv5_3_de'], 512, 3, pad=1)
    net['uconv5_3_de'].add_param(net['uconv5_3_de'].W, net['uconv5_3_de'].W.get_value().shape, trainable=False)
    net['uconv5_3_de'].add_param(net['uconv5_3_de'].b, net['uconv5_3_de'].b.get_value().shape, trainable=False)
    print "uconv5_3_de: {}".format(net['uconv5_3_de'].output_shape[1:])

    net['uconv5_2_de'] = ConvLayer(net['uconv5_3_de'], 512, 3, pad=1)
    net['uconv5_2_de'].add_param(net['uconv5_2_de'].W, net['uconv5_2_de'].W.get_value().shape, trainable=False)
    net['uconv5_2_de'].add_param(net['uconv5_2_de'].b, net['uconv5_2_de'].b.get_value().shape, trainable=False)
    print "uconv5_2_de: {}".format(net['uconv5_2_de'].output_shape[1:])

    net['uconv5_1_de'] = ConvLayer(net['uconv5_2_de'], 512, 3, pad=1)
    net['uconv5_1_de'].add_param(net['uconv5_1_de'].W, net['uconv5_1_de'].W.get_value().shape, trainable=False)
    net['uconv5_1_de'].add_param(net['uconv5_1_de'].b, net['uconv5_1_de'].b.get_value().shape, trainable=False)
    print "uconv5_1_de: {}".format(net['uconv5_1_de'].output_shape[1:])

    net['upool4_de'] = Upscale2DLayer(net['uconv5_1_de'], scale_factor=2)
    print "upool4_de: {}".format(net['upool4_de'].output_shape[1:])

    net['uconv4_3_de'] = ConvLayer(net['upool4_de'], 512, 3, pad=1)
    net['uconv4_3_de'].add_param(net['uconv4_3_de'].W, net['uconv4_3_de'].W.get_value().shape, trainable=False)
    net['uconv4_3_de'].add_param(net['uconv4_3_de'].b, net['uconv4_3_de'].b.get_value().shape, trainable=False)
    print "uconv4_3_de: {}".format(net['uconv4_3_de'].output_shape[1:])

    net['uconv4_2_de'] = ConvLayer(net['uconv4_3_de'], 512, 3, pad=1)
    net['uconv4_2_de'].add_param(net['uconv4_2_de'].W, net['uconv4_2_de'].W.get_value().shape, trainable=False)
    net['uconv4_2_de'].add_param(net['uconv4_2_de'].b, net['uconv4_2_de'].b.get_value().shape, trainable=False)
    print "uconv4_2_de: {}".format(net['uconv4_2_de'].output_shape[1:])

    net['uconv4_1_de'] = ConvLayer(net['uconv4_2_de'], 512, 3, pad=1)
    net['uconv4_1_de'].add_param(net['uconv4_1_de'].W, net['uconv4_1_de'].W.get_value().shape, trainable=False)
    net['uconv4_1_de'].add_param(net['uconv4_1_de'].b, net['uconv4_1_de'].b.get_value().shape, trainable=False)
    print "uconv4_1_de: {}".format(net['uconv4_1_de'].output_shape[1:])

    net['upool3_de'] = Upscale2DLayer(net['uconv4_1_de'], scale_factor=2)
    print "upool3_de: {}".format(net['upool3_de'].output_shape[1:])

    net['uconv3_3_de'] = ConvLayer(net['upool3_de'], 256, 3, pad=1)
    net['uconv3_3_de'].add_param(net['uconv3_3_de'].W, net['uconv3_3_de'].W.get_value().shape, trainable=False)
    net['uconv3_3_de'].add_param(net['uconv3_3_de'].b, net['uconv3_3_de'].b.get_value().shape, trainable=False)
    print "uconv3_3_de: {}".format(net['uconv3_3_de'].output_shape[1:])

    net['uconv3_2_de'] = ConvLayer(net['uconv3_3_de'], 256, 3, pad=1)
    net['uconv3_2_de'].add_param(net['uconv3_2_de'].W, net['uconv3_2_de'].W.get_value().shape, trainable=False)
    net['uconv3_2_de'].add_param(net['uconv3_2_de'].b, net['uconv3_2_de'].b.get_value().shape, trainable=False)
    print "uconv3_2_de: {}".format(net['uconv3_2_de'].output_shape[1:])

    net['uconv3_1_de'] = ConvLayer(net['uconv3_2_de'], 256, 3, pad=1)
    net['uconv3_1_de'].add_param(net['uconv3_1_de'].W, net['uconv3_1_de'].W.get_value().shape, trainable=False)
    net['uconv3_1_de'].add_param(net['uconv3_1_de'].b, net['uconv3_1_de'].b.get_value().shape, trainable=False)
    print "uconv3_1_de: {}".format(net['uconv3_1_de'].output_shape[1:])

    net['upool2_de'] = Upscale2DLayer(net['uconv3_1_de'], scale_factor=2)
    print "upool2_de: {}".format(net['upool2_de'].output_shape[1:])

    net['uconv2_2_de'] = ConvLayer(net['upool2_de'], 128, 3, pad=1)
    net['uconv2_2_de'].add_param(net['uconv2_2_de'].W, net['uconv2_2_de'].W.get_value().shape, trainable=False)
    net['uconv2_2_de'].add_param(net['uconv2_2_de'].b, net['uconv2_2_de'].b.get_value().shape, trainable=False)
    print "uconv2_2_de: {}".format(net['uconv2_2_de'].output_shape[1:])

    net['uconv2_1_de'] = ConvLayer(net['uconv2_2_de'], 128, 3, pad=1)
    net['uconv2_1_de'].add_param(net['uconv2_1_de'].W, net['uconv2_1_de'].W.get_value().shape, trainable=False)
    net['uconv2_1_de'].add_param(net['uconv2_1_de'].b, net['uconv2_1_de'].b.get_value().shape, trainable=False)
    print "uconv2_1_de: {}".format(net['uconv2_1_de'].output_shape[1:])

    net['upool1_de'] = Upscale2DLayer(net['uconv2_1_de'], scale_factor=2)
    print "upool1_de: {}".format(net['upool1_de'].output_shape[1:])

    net['uconv1_2_de'] = ConvLayer(net['upool1_de'], 64, 3, pad=1,)
    net['uconv1_2_de'].add_param(net['uconv1_2_de'].W, net['uconv1_2_de'].W.get_value().shape, trainable=False)
    net['uconv1_2_de'].add_param(net['uconv1_2_de'].b, net['uconv1_2_de'].b.get_value().shape, trainable=False)
    print "uconv1_2_de: {}".format(net['uconv1_2_de'].output_shape[1:])

    net['uconv1_1_de'] = ConvLayer(net['uconv1_2_de'], 64, 3, pad=1)
    net['uconv1_1_de'].add_param(net['uconv1_1_de'].W, net['uconv1_1_de'].W.get_value().shape, trainable=False)
    net['uconv1_1_de'].add_param(net['uconv1_1_de'].b, net['uconv1_1_de'].b.get_value().shape, trainable=False)
    print "uconv1_1_de: {}".format(net['uconv1_1_de'].output_shape[1:])

    net['output'] = ConvLayer(net['uconv1_1_de'], 1, 1, pad=0,nonlinearity=sigmoid)
    print "output: {}".format(net['output'].output_shape[1:])

    return net


def build(net, input_height, input_width):

    encoder_de = build_encoder(net,input_height, input_width)
    autoencoder = build_decoder(encoder_de)
  #  load_weights(autoencoder['output'], path='gen_', epochtoload=90)
    return autoencoder
