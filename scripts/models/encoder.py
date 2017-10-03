from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Upscale2DLayer, InputLayer
from lasagne.layers import ScaleLayer, BiasLayer 
from lasagne.nonlinearities import sigmoid,tanh
import lasagne
import cPickle
import vgg16
from constants import PATH_TO_VGG16_WEIGHTS

def build_vgg16(inputHeight, inputWidth, input_var):
    """
    Build the encoder model
    The input is the inputHeight * inputWidth * 4 image + saliency.
    The output is 512,3 filters.
    There is no fully-connected layers.
    This vgg16 model is trainable.
    
    """
    net = {'input': InputLayer((None, 4, inputHeight, inputWidth), input_var=input_var)}

    print "Input: {}".format(net['input'].output_shape[1:])

    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    print "conv1_1: {}".format(net['conv1_1'].output_shape[1:])

    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    print "conv1_2: {}".format(net['conv1_2'].output_shape[1:])

    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    print "pool1: {}".format(net['pool1'].output_shape[1:])

    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    print "conv2_1: {}".format(net['conv2_1'].output_shape[1:])

    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    print "conv2_2: {}".format(net['conv2_2'].output_shape[1:])

    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    print "pool2: {}".format(net['pool2'].output_shape[1:])

    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    print "conv3_1: {}".format(net['conv3_1'].output_shape[1:])

    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    print "conv3_2: {}".format(net['conv3_2'].output_shape[1:])

    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    print "conv3_3: {}".format(net['conv3_3'].output_shape[1:])

    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    print "pool3: {}".format(net['pool3'].output_shape[1:])

    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    print "conv4_1: {}".format(net['conv4_1'].output_shape[1:])

    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    print "conv4_2: {}".format(net['conv4_2'].output_shape[1:])

    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    print "conv4_3: {}".format(net['conv4_3'].output_shape[1:])

    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    print "pool4: {}".format(net['pool4'].output_shape[1:])

    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    print "conv5_1: {}".format(net['conv5_1'].output_shape[1:])

    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    print "conv5_2: {}".format(net['conv5_2'].output_shape[1:])

    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    print "conv5_3: {}".format(net['conv5_3'].output_shape[1:])

    return net



# def set_pretrained_weights(net, path_to_model_weights=PATH_TO_VGG16_WEIGHTS):
#     # Set out weights
#     vgg16 = cPickle.load(open(path_to_model_weights))
#     num_elements_to_set = 26  # Number of W and b elements for the first convolutional layers
#     lasagne.layers.set_all_param_values(net['conv5_3'], vgg16['param values'][:num_elements_to_set])

def set_pretrained_weights(net, path_to_model_weights=PATH_TO_VGG16_WEIGHTS):
    # Set out weights
    vgg16 = cPickle.load(open(path_to_model_weights))
    print ("Loading vgg16 weights ...")
    num_elements_to_set = 26  # Number of W and b elements for the first convolutional layers
    layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 
               'conv3_1', 'conv3_2', 'conv3_3', 
                'conv4_1', 'conv4_2', 'conv4_3', 
                 'conv5_1', 'conv5_2','conv5_3']
    layers = [ net[k].get_params() for k in layers]

    for i in range(len(layers)):
        if len(layers[i]) == 2:
            layers[i][0].set_value(vgg16['param values'][i*2])
            layers[i][1].set_value(vgg16['param values'][i*2+1])

def build_encoder(input_height, input_width, input_var):
    encoder = build_vgg16(input_height, input_width, input_var)
    #set_pretrained_weights(encoder)
    return encoder


def build_decoder(net):
    net['uconv5_3']= ConvLayer(net['conv5_3'], 512, 3, pad=1)
    print "uconv5_3: {}".format(net['uconv5_3'].output_shape[1:])

    net['uconv5_2'] = ConvLayer(net['uconv5_3'], 512, 3, pad=1)
    print "uconv5_2: {}".format(net['uconv5_2'].output_shape[1:])

    net['uconv5_1'] = ConvLayer(net['uconv5_2'], 512, 3, pad=1)
    print "uconv5_1: {}".format(net['uconv5_1'].output_shape[1:])

    net['upool4'] = Upscale2DLayer(net['uconv5_1'], scale_factor=2)
    print "upool4: {}".format(net['upool4'].output_shape[1:])

    net['uconv4_3'] = ConvLayer(net['upool4'], 512, 3, pad=1)
    print "uconv4_3: {}".format(net['uconv4_3'].output_shape[1:])

    net['uconv4_2'] = ConvLayer(net['uconv4_3'], 512, 3, pad=1)
    print "uconv4_2: {}".format(net['uconv4_2'].output_shape[1:])

    net['uconv4_1'] = ConvLayer(net['uconv4_2'], 512, 3, pad=1)
    print "uconv4_1: {}".format(net['uconv4_1'].output_shape[1:])

    net['upool3'] = Upscale2DLayer(net['uconv4_1'], scale_factor=2)
    print "upool3: {}".format(net['upool3'].output_shape[1:])

    net['uconv3_3'] = ConvLayer(net['upool3'], 256, 3, pad=1)
    print "uconv3_3: {}".format(net['uconv3_3'].output_shape[1:])

    net['uconv3_2'] = ConvLayer(net['uconv3_3'], 256, 3, pad=1)
    print "uconv3_2: {}".format(net['uconv3_2'].output_shape[1:])

    net['uconv3_1'] = ConvLayer(net['uconv3_2'], 256, 3, pad=1)
    print "uconv3_1: {}".format(net['uconv3_1'].output_shape[1:])

    net['upool2'] = Upscale2DLayer(net['uconv3_1'], scale_factor=2)
    print "upool2: {}".format(net['upool2'].output_shape[1:])

    net['uconv2_2'] = ConvLayer(net['upool2'], 128, 3, pad=1)
    print "uconv2_2: {}".format(net['uconv2_2'].output_shape[1:])

    net['uconv2_1'] = ConvLayer(net['uconv2_2'], 128, 3, pad=1)
    print "uconv2_1: {}".format(net['uconv2_1'].output_shape[1:])

    net['upool1'] = Upscale2DLayer(net['uconv2_1'], scale_factor=2)
    print "upool1: {}".format(net['upool1'].output_shape[1:])

    net['uconv1_2'] = ConvLayer(net['upool1'], 64, 3, pad=1,)
    print "uconv1_2: {}".format(net['uconv1_2'].output_shape[1:])

    net['uconv1_1'] = ConvLayer(net['uconv1_2'], 64, 3, pad=1)
    print "uconv1_1: {}".format(net['uconv1_1'].output_shape[1:])

    net['output_encoder'] = ConvLayer(net['uconv1_1'], 3, 1, pad=0,nonlinearity=tanh)
    print "output_encoder: {}".format(net['output_encoder'].output_shape[1:])

    net['output_encoder_bias'] = BiasLayer(net['output_encoder'], b=lasagne.init.Constant(1))
    print "output_encoder_bias: {}".format(net['output_encoder_bias'].output_shape[1:])
    net['output_encoder_scaled'] = ScaleLayer(net['output_encoder_bias'], scales=lasagne.init.Constant(127.5))
    print "output_encoder_scaled: {}".format(net['output_encoder_scaled'].output_shape[1:])

    return net


def build(input_height, input_width, input_var):
    encoder = build_encoder(input_height, input_width, input_var)
    generator = build_decoder(encoder)
    return generator
