import numpy as np
import os
import cv2
import theano
import lasagne
import random
import glob
from tqdm import tqdm
from constants import HOME_DIR, WEIGHTS_DIR,TEST_DIR

def merge(images, saliency, size):
    """merge images to one image"""
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, (image, sali) in enumerate(zip(images,saliency)):
        sal = cv2.cvtColor(sali, cv2.COLOR_GRAY2RGB)
        im = image
        i = idx % size[1]
        j = 2 * (idx // size[1])
        img[j*h:j*h+h, i*w:i*w+w, :] = sal
        img[j*h+h:j*h+h+h, i*w:i*w+w, :] = im

    return img

def transform(im):
    """for tanh, normalize input imgs"""
    return im/127.5 - 1

def iterate_minibatches(inputs, batchsize, shuffle=False):
    indices = range(len(inputs))
    while True :
        if shuffle :
            random.shuffle(indices)
        for start_idx in range(0, len(inputs)- batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = range(start_idx, start_idx + batch_size)
            yield [inputs[i] for i in excerpt]

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
              'uconv1_1_de', 'output']

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
    with np.load(WEIGHTS_DIR + path + "modelWeights{:04d}.npz".format(epochtoload)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)


def predict(model, image_stimuli, saliency_stimuli, num_epoch=None, name=None, path_output_imgs=None):

    size = (image_stimuli.shape[1], image_stimuli.shape[0])
    blur_size = 5

    if image_stimuli.shape[:2] != (model.inputHeight, model.inputWidth):
        image_stimuli = cv2.resize(image_stimuli, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)
    if saliency_stimuli.shape[:2] != (model.inputHeight, model.inputWidth):
        saliency_stimuli = cv2.resize(saliency_stimuli, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)

    #blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    blob = np.zeros((1, 4, model.inputHeight, model.inputWidth), theano.config.floatX)
    
    input_blob = np.append(transform(image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1)), transform(saliency_stimuli.astype(theano.config.floatX).reshape(1, model.inputHeight, model.inputWidth)), axis=0)
   # input_blob = image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1)
    blob[0, ...] = input_blob

    result = np.squeeze(model.predictFunction(blob))
    #print(result)
    #guided_image = result
    guided_image = (result * 255).astype(np.uint8)
    #guided_image = ( (result+1) * 127.5).astype(np.uint8)
    #guided_image = guided_image.transpose(1, 2, 0)
    #guided_image = cv2.cvtColor(guided_image, cv2.COLOR_RGB2BGR)
    # resize back to original size
    guided_image = cv2.resize(guided_image, size, interpolation=cv2.INTER_CUBIC)
    guided_image = cv2.GaussianBlur(guided_image, (blur_size, blur_size), 0)
    # clip again
    #
    guided_image = np.clip(guided_image, 0, 255)
    #print(guided_image.shape)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite('./' + path_output_imgs + '/validationRandomSaliencyPred_{:04d}.png'.format(num_epoch), guided_image)
    else:
        cv2.imwrite(os.path.join(path_output_imgs, name + '.jpg'), guided_image)

def test_batch(path_to_images, path_to_saliency, name, model=None):
    list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(path_to_saliency, '*'))]
    # Load Data
    list_img_files.sort()
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
    batch_img = np.empty((0,3,model.inputHeight, model.inputWidth), dtype=theano.config.floatX)
    batch_sal = np.empty((0,1,model.inputHeight, model.inputWidth), dtype=theano.config.floatX)

    for curr_file in tqdm(list_img_files, ncols=20):
        #img = np.random.random((model.inputHeight,model.inputWidth,3))*255
        img = cv2.cvtColor(cv2.imread(os.path.join(path_to_images, curr_file + '.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        sal = cv2.imread(os.path.join(path_to_saliency, curr_file + '.jpg'),cv2.IMREAD_GRAYSCALE)
        ## resize to model input
        if img.shape[:2] != (model.inputHeight, model.inputWidth):
            img = cv2.resize(img, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)
        if sal.shape[:2] != (model.inputHeight, model.inputWidth):
            sal = cv2.resize(sal, (model.inputWidth, model.inputHeight), interpolation=cv2.INTER_AREA)
        
    
        img = img.transpose(2,0,1)
        batch_img = np.append(batch_img, img.reshape((1,3,model.inputHeight,model.inputWidth)), axis=0)
        batch_sal = np.append(batch_sal, sal.reshape((1,1,model.inputHeight,model.inputWidth)), axis=0)

    #blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)

    predict_batch(model=model, image_stimuli=batch_img, saliency_stimuli=batch_sal, name=name)

def predict_batch(model, image_stimuli, saliency_stimuli, num_epoch=None, name=None):

    size = (image_stimuli.shape[1], image_stimuli.shape[0])
    blur_size = 5
    batch_size =image_stimuli.shape[0]
    #blob = np.zeros((1, 3, model.inputHeight, model.inputWidth), theano.config.floatX)
    blob = np.zeros((batch_size, 4, model.inputHeight, model.inputWidth), theano.config.floatX)
    
    input_blob = np.append(transform(image_stimuli.astype(theano.config.floatX)), transform(saliency_stimuli.astype(theano.config.floatX)), axis=1)
   # input_blob = image_stimuli.astype(theano.config.floatX).transpose(2, 0, 1)
    blob[:] = input_blob

    #result = np.squeeze(model.autoencoder.predictFunction(blob))
    result = np.squeeze(model.autoencoder.predictFunction(blob))
    #print(result)
    guided_image = ((result+1)*127.5).astype(np.uint8)
    #guided_image = (result * 255).astype(np.uint8)
    guided_image = np.asarray([guided_image[i].transpose(1,2,0) for i in range(guided_image.shape[0])], dtype=theano.config.floatX)
    guided_image = np.asarray([cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in guided_image], dtype=theano.config.floatX)
    # norm transform back
    # resize back to original size
    #guided_image = cv2.resize(guided_image, size, interpolation=cv2.INTER_CUBIC)
    #guided_image = cv2.GaussianBlur(guided_image, (blur_size, blur_size), 0)
    # clip again
    #

    guided_image = np.clip(guided_image, 0, 255)
    sali_ori = np.asarray([saliency_stimuli[i].transpose(1,2,0) for i in range(saliency_stimuli.shape[0])], dtype=theano.config.floatX)

    merge_image = merge(guided_image,sali_ori,(6,5))
    #print(guided_image.shape)
    if name is None:
        # When we use for testing, there is no file name provided.
        cv2.imwrite(TEST_DIR + '/validationRandomSaliencyPred_{:04d}.png'.format(num_epoch), merge_image)
    else:
        cv2.imwrite(os.path.join(TEST_DIR, name + '.jpg'), merge_image)






