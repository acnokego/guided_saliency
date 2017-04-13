import numpy as np
import os
import cv2
import theano
import lasagne
from constants import HOME_DIR


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def load_weights(net, path, epochtoload):
    """
    Load a pretrained model
    :param epochtoload: epoch to load
    :param net: model object
    :param path: path of the weights to be set
    """
    print ("Loading decoder weights ...")
    with np.load(HOME_DIR + path + "modelWeights{:04d}.npz".format(epochtoload)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    layers = ['conv1_1_de', 'conv1_2_de', 'conv2_1_de', 'conv2_2_de', 
              'conv3_1_de', 'conv3_2_de', 'conv3_3_de', 
              'conv4_1_de', 'conv4_2_de', 'conv4_3_de', 
              'conv5_1_de', 'conv5_2_de', 'conv5_3_de', 
              'uconv5_3_de', 'uconv5_2_de', 'uconv5_1_de', 'uconv4_3_de',
              'uconv4_2_de', 'uconv4_1_de', 'uconv3_3_de', 'uconv3_2_de',
              'uconv3_1_de', 'uconv2_2_de', 'uconv2_1_de', 'uconv1_2_de',
              'uconv1_1_de']

    layers = [ net[k].get_params() for k in layers]

    for i in range(len(layers)):
        if len(layers[i]) == 2:
            layers[i][0].set_value(param_values[i*2])
            layers[i][1].set_value(param_values[i*2+1])
    #lasagne.layers.set_all_param_values(net, param_values)
def load_weights_test(net, path, epochtoload):
    """
    Load a pretrained model
    :param epochtoload: epoch to load
    :param net: model object
    :param path: path of the weights to be set
    """
    print("Loading encoder weights...")
    with np.load(HOME_DIR + path + "modelWeights{:04d}.npz".format(epochtoload)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)


def predict(model, image_stimuli, saliency_stimuli, num_epoch=None, name=None, path_output_imgs=None):

    size = (image_stimuli.shape[1], image_stimuli.shape[0])
    blur_size = 5

    if image_stimuli.shape[:2] != (model.inputHeight, model.inputWidth):
        image_stimuli = cv2.resize(image_stimuli, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)
    if saliency_stimuli.shape[:2] != (model.inputHeight, model.inputWidth):
        saliency_stimuli = cv2.resize(saliency_stimuli, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)

    blob = np.zeros((1, 4, model.inputHeight, model.inputWidth), theano.config.floatX)
    
    input_blob = np.append(image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1), saliency_stimuli.astype(theano.config.floatX).reshape(1, model.inputHeight, model.inputWidth), axis=0)
    blob[0, ...] = (input_blob)

    result = np.squeeze(model.predictFunction(blob))

    guided_image = (result * 255).astype(np.uint8)
    guided_image = guided_image.transpose(1, 2, 0)
    # resize back to original size
    #guided_image = cv2.resize(guided_image, size, interpolation=cv2.INTER_CUBIC)
    #guided_image = cv2.GaussianBlur(guided_image, (blur_size, blur_size), 0)
    # clip again
    #
    guided_image = np.clip(guided_image, 0, 255)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite('./' + path_output_imgs + '/validationRandomSaliencyPred_{:04d}.png'.format(num_epoch), guided_image)
    else:
        cv2.imwrite(os.path.join(path_output_imgs, name + '.jpg'), guided_image)


