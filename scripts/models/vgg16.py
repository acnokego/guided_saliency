# VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
# License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer
from layers import RGBtoBGRLayer

def build(net, inputHeight, inputWidth, connect=True, input_var=None):
    """
    Bulid only Convolutional part of the VGG-16 Layer model, all fully connected layers are removed.
    First 3 group of ConvLayers are fixed (not trainable).

    :param input_layer: Input layer of the network.
    :return: Dictionary that contains all layers.
    """

 #   net = {'input': InputLayer((None, 3, inputHeight, inputWidth), input_var=input_var)}
 #   print "Input: {}".format(net['input'].output_shape[1:])
    if connect:
        net['bgr'] = RGBtoBGRLayer(net['output_encoder_scaled'])
    else :
        #print ("build vgg net for content loss")
        net = {'input': InputLayer((None, 3, inputHeight, inputWidth), input_var=input_var)}
        net['bgr'] = RGBtoBGRLayer(net['input'])
        print "Input: {}".format(net['input'].output_shape[1:])
        

    net['conv1_1_de'] = ConvLayer(net['bgr'], 64, 3, pad=1, flip_filters=False)
    net['conv1_1_de'].add_param(net['conv1_1_de'].W, net['conv1_1_de'].W.get_value().shape, trainable=False)
    net['conv1_1_de'].add_param(net['conv1_1_de'].b, net['conv1_1_de'].b.get_value().shape, trainable=False)
    print "conv1_1_de: {}".format(net['conv1_1_de'].output_shape[1:])

    net['conv1_2_de'] = ConvLayer(net['conv1_1_de'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2_de'].add_param(net['conv1_2_de'].W, net['conv1_2_de'].W.get_value().shape, trainable=False)
    net['conv1_2_de'].add_param(net['conv1_2_de'].b, net['conv1_2_de'].b.get_value().shape, trainable=False)
    print "conv1_2_de: {}".format(net['conv1_2_de'].output_shape[1:])

    net['pool1_de'] = PoolLayer(net['conv1_2_de'], 2)
    print "pool1_de: {}".format(net['pool1_de'].output_shape[1:])

    net['conv2_1_de'] = ConvLayer(net['pool1_de'], 128, 3, pad=1, flip_filters=False)
    net['conv2_1_de'].add_param(net['conv2_1_de'].W, net['conv2_1_de'].W.get_value().shape, trainable=False)
    net['conv2_1_de'].add_param(net['conv2_1_de'].b, net['conv2_1_de'].b.get_value().shape, trainable=False)
    print "conv2_1_de: {}".format(net['conv2_1_de'].output_shape[1:])

    net['conv2_2_de'] = ConvLayer(net['conv2_1_de'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2_de'].add_param(net['conv2_2_de'].W, net['conv2_2_de'].W.get_value().shape, trainable=False)
    net['conv2_2_de'].add_param(net['conv2_2_de'].b, net['conv2_2_de'].b.get_value().shape, trainable=False)
    print "conv2_2_de: {}".format(net['conv2_2_de'].output_shape[1:])

    net['pool2_de'] = PoolLayer(net['conv2_2_de'], 2)
    print "pool2_de: {}".format(net['pool2_de'].output_shape[1:])

    net['conv3_1_de'] = ConvLayer(net['pool2_de'], 256, 3, pad=1, flip_filters=False)
    net['conv3_1_de'].add_param(net['conv3_1_de'].W, net['conv3_1_de'].W.get_value().shape, trainable=False)
    net['conv3_1_de'].add_param(net['conv3_1_de'].b, net['conv3_1_de'].b.get_value().shape, trainable=False)
    print "conv3_1_de: {}".format(net['conv3_1_de'].output_shape[1:])

    net['conv3_2_de'] = ConvLayer(net['conv3_1_de'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2_de'].add_param(net['conv3_2_de'].W, net['conv3_2_de'].W.get_value().shape, trainable=False)
    net['conv3_2_de'].add_param(net['conv3_2_de'].b, net['conv3_2_de'].b.get_value().shape, trainable=False)
    print "conv3_2_de: {}".format(net['conv3_2_de'].output_shape[1:])

    net['conv3_3_de'] = ConvLayer(net['conv3_2_de'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3_de'].add_param(net['conv3_3_de'].W, net['conv3_3_de'].W.get_value().shape, trainable=False)
    net['conv3_3_de'].add_param(net['conv3_3_de'].b, net['conv3_3_de'].b.get_value().shape, trainable=False)
    print "conv3_3_de: {}".format(net['conv3_3_de'].output_shape[1:])

    net['pool3_de'] = PoolLayer(net['conv3_3_de'], 2)
    print "pool3_de: {}".format(net['pool3_de'].output_shape[1:])

    net['conv4_1_de'] = ConvLayer(net['pool3_de'], 512, 3, pad=1, flip_filters=False)
    net['conv4_1_de'].add_param(net['conv4_1_de'].W, net['conv4_1_de'].W.get_value().shape, trainable=False)
    net['conv4_1_de'].add_param(net['conv4_1_de'].b, net['conv4_1_de'].b.get_value().shape, trainable=False)
    print "conv4_1_de: {}".format(net['conv4_1_de'].output_shape[1:])

    net['conv4_2_de'] = ConvLayer(net['conv4_1_de'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2_de'].add_param(net['conv4_2_de'].W, net['conv4_2_de'].W.get_value().shape, trainable=False)
    net['conv4_2_de'].add_param(net['conv4_2_de'].b, net['conv4_2_de'].b.get_value().shape, trainable=False)
    print "conv4_2_de: {}".format(net['conv4_2_de'].output_shape[1:])

    net['conv4_3_de'] = ConvLayer(net['conv4_2_de'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3_de'].add_param(net['conv4_3_de'].W, net['conv4_3_de'].W.get_value().shape, trainable=False)
    net['conv4_3_de'].add_param(net['conv4_3_de'].b, net['conv4_3_de'].b.get_value().shape, trainable=False)
    print "conv4_3_de: {}".format(net['conv4_3_de'].output_shape[1:])

    net['pool4_de'] = PoolLayer(net['conv4_3_de'], 2)
    print "pool4_de: {}".format(net['pool4_de'].output_shape[1:])

    net['conv5_1_de'] = ConvLayer(net['pool4_de'], 512, 3, pad=1, flip_filters=False)
    net['conv5_1_de'].add_param(net['conv5_1_de'].W, net['conv5_1_de'].W.get_value().shape, trainable=False)
    net['conv5_1_de'].add_param(net['conv5_1_de'].b, net['conv5_1_de'].b.get_value().shape, trainable=False)
    print "conv5_1_de: {}".format(net['conv5_1_de'].output_shape[1:])

    net['conv5_2_de'] = ConvLayer(net['conv5_1_de'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2_de'].add_param(net['conv5_2_de'].W, net['conv5_2_de'].W.get_value().shape, trainable=False)
    net['conv5_2_de'].add_param(net['conv5_2_de'].b, net['conv5_2_de'].b.get_value().shape, trainable=False)
    print "conv5_2_de: {}".format(net['conv5_2_de'].output_shape[1:])

    net['conv5_3_de'] = ConvLayer(net['conv5_2_de'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3_de'].add_param(net['conv5_3_de'].W, net['conv5_3_de'].W.get_value().shape, trainable=False)
    net['conv5_3_de'].add_param(net['conv5_3_de'].b, net['conv5_3_de'].b.get_value().shape, trainable=False)
    print "conv5_3_de: {}".format(net['conv5_3_de'].output_shape[1:])

    return net

