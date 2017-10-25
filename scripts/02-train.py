#   Two mode of training available:
#       - BCE: CNN training, NOT Adversarial Training here. Only learns the generator network.
#       - SALGAN: Adversarial Training. Updates weights for both Generator and Discriminator.
#   The training used data previously  processed using "01-data_preocessing.py"
import os
import numpy as np
import sys
import cPickle as pickle
import random
import cv2
import theano
import theano.tensor as T
import lasagne

from tqdm import tqdm
from constants import *
from models.autoencoder import autoencoder
from models.biGAN import biGAN
from utils import *
import models.style_transfer as style
import matplotlib.pyplot as plt


flag = str(sys.argv[1])

def autoencoder_batch_iterator(model, train_data):
    num_epochs = 101
    nr_batches_train = int(len(train_data) / model.batch_size)
    n_updates = 1
    losses = np.zeros((1,2))
    for current_epoch in tqdm(range(num_epochs), ncols=20):

        l2_cost = 0.
        content_cost = 0.

        random.shuffle(train_data)

        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue
            batch_input = np.asarray([np.append(x.image.data.astype(theano.config.floatX).transpose(2, 0, 1),
                                      x.saliency.data.astype(theano.config.floatX).reshape(1,INPUT_SIZE[1],
                                    INPUT_SIZE[0]),axis=0) for x in currChunk],
                                     dtype=theano.config.floatX)
            #BGR input image 
            batch_input_imgs = np.asarray([style.prep_image(x.image.data).astype(theano.config.floatX) 
                                           for x in currChunk], dtype=theano.config.floatX)
            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],
                                      dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)

            # train generator with one batch and discriminator with next batch
            if n_updates % 2 == 0:
                l2_loss = model.G_trainFunction(batch_input, batch_output)
                l2_cost += l2_loss
            else:
               # G_obj, D_obj, G_cost = model.D_trainFunction(batch_input, batch_output)
                c_loss = model.content_trainFunction(batch_input, batch_input_imgs)
                content_cost += c_loss

            n_updates += 1

        l2_cost /= nr_batches_train
        content_cost /= nr_batches_train

        # Save weights every 3 epoch
        if current_epoch % 3 == 0:
            np.savez('/' + DIR_TO_SAVE + '/auto_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output_encoder']))
            #predict(model=model, image_stimuli=validation_sample, numEpoch=current_epoch, pathOutputMaps=DIR_TO_SAVE)
        print 'Epoch:', current_epoch, ' train_loss->', (l2_cost, content_cost)
        tp_losses = np.array([[l2_cost, content_cost]])
        losses = np.append(losses, tp_losses, axis=0)

       # if current_epoch % 10 == 0:
        #    np.savez('./loss_0.0001_toy_center_1layer.npz', losses)


def autoencoder_batch_iterator_separate(model, train_data):
    num_epochs = 100
    nr_batches_train = int(len(train_data) / model.batch_size)
    n_updates = 1

    
    loss = np.empty(0)
    for current_epoch in tqdm(range(num_epochs), ncols=20):

        l2_cost = 0.

        random.shuffle(train_data)

        for currChunk in chunks(train_data, model.batch_size):

            if len(currChunk) != model.batch_size:
                continue
            batch_input = np.asarray([transform(np.append(x.image.data.astype(theano.config.floatX).transpose(2, 0, 1),
                                      x.saliency.data.astype(theano.config.floatX).reshape(1,INPUT_SIZE[1],
                                      INPUT_SIZE[0]),axis=0)) for x in currChunk],
                                     dtype=theano.config.floatX)
            #batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1)
            #                         for x in currChunk],
            #                         dtype=theano.config.floatX)
            #batch_input_imgs = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2,0,1) 
            #                               for x in currChunk], dtype=theano.config.floatX)
            batch_input_imgs = np.asarray([(x.image.data).astype(theano.config.floatX).transpose(2,0,1)
                                           for x in currChunk], dtype=theano.config.floatX)
            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],
                                      dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)

            # train generator with one batch and discriminator with next batch
            #l2_loss = model.G_trainFunction(batch_input, batch_input_imgs)
            l2_loss = model.G_trainFunction(batch_input, batch_output, batch_input_imgs)
            print 'Iter:', n_updates, ' train_loss->', (l2_loss)
            l2_cost += l2_loss

            n_updates += 1

        l2_cost /= nr_batches_train
        loss = np.append(loss, l2_cost)
        if current_epoch % 3 == 0:
            np.savez('/' + DIR_TO_SAVE + '/auto_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.encoder['output_encoder']))
        print 'Epoch:', current_epoch, ' train_loss->', (l2_cost)
        if current_epoch % 10 == 0:
            np.savez(LOSS_TO_SAVE, loss)

def biGAN_batch_iterator(model, train_data, real_data):
    num_epochs = 30001
    nr_batches_train = int(len(train_data) / model.batch_size)
    n_updates = 1
    losses = np.zeros((1,5))
    batches_D = iterate_minibatches(train_data, model.batch_size, True)
    batches_G = iterate_minibatches(train_data, model.batch_size, True)
    batches_real = iterate_minibatches(real_data, model.batch_size, True)
    generator_updates = 1

    decay_thre = 0.6
    decay_epochs = 10001
    pi = np.zeros((num_epochs))
    pi[:int(decay_epochs*decay_thre)] = np.linspace(0.25, 0, int(decay_epochs*decay_thre))
    for current_epoch in tqdm(range(num_epochs), ncols=20):

        #auto_cost = 0.
        G_cost = []
        D_cost = []

        #else:
        #    discri_run = 5
        
        discri_run = 5         
        for _ in range(discri_run):
            batch = next(batches_D)
            batch_real = next(batches_real)
            noise_image = np.random.random((128,128,3))*255
            
            batch_input = np.asarray([np.append(x.image.data.astype(theano.config.floatX).transpose(2, 0, 1),
                                      x.saliency.data.astype(theano.config.floatX).reshape(1,INPUT_SIZE[1],
                                    INPUT_SIZE[0]),axis=0) for x in batch],
                                     dtype=theano.config.floatX)
            
            
            batch_real_input = np.asarray([transform(np.append(x.image.data.astype(theano.config.floatX).transpose(2, 0, 1),
                                      x.saliency.data.astype(theano.config.floatX).reshape(1,INPUT_SIZE[1],
                                    INPUT_SIZE[0]),axis=0)) for x in batch_real],
                                     dtype=theano.config.floatX)
            
            '''
            batch_real_input = np.asarray([transform(x.image.data.astype(theano.config.floatX).transpose(2, 0, 1))
                                           for x in batch_real],
                                        dtype=theano.config.floatX)
            '''
            #   Generate noise label to discriminator 
            #true_label_noise, false_label_noise = model.generateNoise(pi=0.1)
            
            #D_cost.append(model.D_train_fn(batch_input, batch_real_input, true_label_noise, false_label_noise))

            ## wGAN
            D_cost.append(model.D_train_fn(batch_input, batch_real_input))
            
                
        batch = next(batches_G)
        #batch_real = next(batches_real)
        noise_image = np.random.random((128,128,3))*255
        '''
        batch_input = np.asarray([(np.append(noise_image.transpose(2, 0, 1),
                                  x.saliency.data.astype(theano.config.floatX).reshape(1,INPUT_SIZE[1],
                                INPUT_SIZE[0]),axis=0)) for x in batch],
                                 dtype=theano.config.floatX)
        '''
        batch_input = np.asarray([(np.append(x.image.data.astype(theano.config.floatX).transpose(2, 0, 1),
                                  x.saliency.data.astype(theano.config.floatX).reshape(1,INPUT_SIZE[1],
                                INPUT_SIZE[0]),axis=0)) for x in batch],
                                 dtype=theano.config.floatX)
        
        #batch_input_imgs = np.asarray([(x.image.data).astype(theano.config.floatX).transpose(2,0,1)
     #                                      for x in batch], dtype=theano.config.floatX)
        batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in         batch],
                                      dtype=theano.config.floatX)
        batch_output = np.expand_dims(batch_output, axis=1)
        
        batch_real_input = np.asarray([transform(np.append(x.image.data.astype(theano.config.floatX).transpose(2, 0, 1),
                                      x.saliency.data.astype(theano.config.floatX).reshape(1,INPUT_SIZE[1],
                                    INPUT_SIZE[0]),axis=0)) for x in batch_real],
                                     dtype=theano.config.floatX)
        
        '''
        batch_real_input = np.asarray([transform(x.image.data.astype(theano.config.floatX).transpose(2, 0, 1))                                                     for x in batch_real],
                                        dtype=theano.config.floatX)
        '''
        G_loss = model.G_train_fn(batch_input, batch_real_input, batch_output)
        #G_loss = model.temp_train(batch_input, batch_output)
        #G_cost.append(G_loss)
        generator_updates += 1 
            



    
        # Save weights every 3 epoch
        if current_epoch % 800 == 0:
            np.savez('/' + DIR_TO_SAVE + '/biGAN_G_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.autoencoder.encoder['output_encoder']))

        if current_epoch % 200 == 0:
            file_to_save = '{:04d}eps'.format(current_epoch)
            test_batch(path_to_images='/home/yuandy/COCO_dataset/train_images/images', path_to_saliency='/home/yuandy/COCO_dataset/temp_test_max1.5', name=file_to_save, model=model)
        print 'Epoch:', current_epoch, ' train_loss->', (G_loss[1], G_loss[2], G_loss[3])
        tp_losses = np.array([[G_loss[0],np.mean(D_cost), G_loss[1], G_loss[2], G_loss[3]]])
        losses = np.append(losses, tp_losses, axis=0)
        #print 'Epoch:', current_epoch, ' train_loss->', (G_loss[0])
        
        if current_epoch % 10 == 0:
            np.savez(LOSS_TO_SAVE, losses)

def train():
    """
    Train both generator and discriminator
    :return:
    """
    # Load data
    print 'Loading training data...'
    with open('/home/yuandy/COCO_dataset/processed_data/128x128/trainData_resize_pool2gs.pickle', 'rb') as f:
    # with open(TRAIN_DATA_DIR, 'rb') as f:
        train_data = pickle.load(f)
    print '-->done!'
    
    print 'Loading real data pair...'
    with open('/home/yuandy/COCO_dataset/processed_data/128x128/realData_resize_pool2gs.pickle', 'rb') as f:
    # with open(TRAIN_DATA_DIR, 'rb') as f:
        real_data = pickle.load(f)
    print '-->done!'
    

    # Create network
    if flag == 'auto':
        model = autoencoder(INPUT_SIZE[0], INPUT_SIZE[1])
        load_weights(model.decoder, path='scripts/gen_', epochtoload=90)
        #load_weights_test(model.encoder['output_encoder'], path='weights_content_new/auto_', epochtoload=21)
        load_weights_test(model.encoder['output_encoder'], path='weights_auto_new_-9/auto_', epochtoload=24)
        autoencoder_batch_iterator_separate(model, train_data)
    elif flag == 'bigan':
        print('ok')
        model = biGAN(INPUT_SIZE)
        load_weights(model.autoencoder.decoder, path='scripts/gen_', epochtoload=90)
        #load_weights_test(model.D, path='weights_bigan_noise_test10/biGAN_D_', epochtoload=29600)
        load_weights_test(model.autoencoder.encoder['output_encoder'], path='weights_auto_new_-9/auto_', epochtoload=24)
        #load_weights_test(model.autoencoder.encoder['output_encoder'], path='weights_content_new/auto_', epochtoload=24)
        #load_weights_test(model.autoencoder.encoder['output_encoder'], path='weights_bigan_noise_test10/biGAN_G_', epochtoload=29600)
        biGAN_batch_iterator(model, train_data, real_data)
    else :
        print ('argument lost...')
if __name__ == "__main__":
    train()
