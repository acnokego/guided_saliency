import theano
import lasagne
import theano.tensor as T
import numpy as np
import style_transfer as style
from autoencoder import autoencoder 
import random

def transform(im):
    return (im/127.5)-1

def build_discriminator(input_var, inputWidth, inputHeight):
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer, batch_norm, dropout)
   # from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer  # override
    from lasagne.nonlinearities import LeakyRectify, sigmoid
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 4, inputHeight, inputWidth), input_var=input_var)
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad=2, nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad=2, nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer
    layer = DenseLayer(layer, 1, nonlinearity=None)
    print ("Discriminator output:", layer.output_shape)
    return layer


class biGAN(object):
    def __init__(self, INPUT_SIZE):
        self.autoencoder = autoencoder(INPUT_SIZE[0],INPUT_SIZE[1])
        self.input_var1 = T.tensor4() # input_var1 = image and saliency to encoder
        self.input_var2 = T.tensor4()# input_var2 = real image and saliency to discriminator
        self.output_var = T.tensor4()
        #self.input_img = T.tensor4()
        self.inputHeight = self.autoencoder.inputHeight
        self.inputWidth = self.autoencoder.inputWidth
        self.batch_size = 16
        self.clip = 0.01
        G_z = lasagne.layers.get_output(self.autoencoder.encoder['output_encoder'], inputs=transform(self.input_var1))
        E_x = self.input_var2 #  input_var = real image and saliency concate
        z = transform(self.input_var1[:,3,:,:])
        z = T.reshape(z, (-1,1,INPUT_SIZE[1],INPUT_SIZE[0]))
        # x = self.input_var2[:,0,:,:]
        self.input_img = self.input_var1[:,:3,:,:]
        #self.output_var = (self.input_var1[:,3,:,:])/255.
        

        """
        for autoencoder

        """
        
        fake_img = lasagne.layers.get_output(self.autoencoder.encoder['output_encoder_scaled'], inputs=transform(self.input_var1))
        prediction = lasagne.layers.get_output(self.autoencoder.decoder['output'], inputs=fake_img)
        ## reconstruction loss
        output_var_pooled = T.signal.pool.pool_2d(self.output_var, (2, 2), 
                                mode="average_exc_pad", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (2, 2), 
                                mode="average_exc_pad", ignore_border=True)
        recon_obj = lasagne.objectives.squared_error(prediction_pooled, 
            output_var_pooled).sum()

        ## content loss

        feature_layers = ['conv4_2_de', 'conv3_1_de', 'conv2_1_de']
   #     feature_layers_2 = 'conv5_2_s'
        input_feature_layers = {k:self.autoencoder.decoder[k] for k in feature_layers}
        generated_feature_layers = {k:self.autoencoder.decoder[k] for k in feature_layers}
        
        outputs = lasagne.layers.get_output(input_feature_layers.values(), 
                                            inputs=self.input_img)
        generated_outputs = lasagne.layers.get_output(generated_feature_layers.values(),
                                            inputs=fake_img)

        ## content loss
        w = float(1.0/len(feature_layers))
        obj2 = 0
        print (w)
        for i in range(len(feature_layers)):
            obj2 = obj2 + w*style.content_loss(generated_outputs[i], outputs[i])
        
        content_obj = obj2
        lam2 = 1e-9

        combine_obj = recon_obj+lam2*content_obj

        """
        for GAN 
        """
        self.D = build_discriminator(self.input_var2, INPUT_SIZE[0], INPUT_SIZE[1])
        self.generator_pair = T.concatenate([G_z, z], axis=1) 
        #self.generator_pair = G_z 
        self.real_pair = self.input_var2
        self.true_label = T.ones((self.batch_size, 1))
        self.false_label = T.ones((self.batch_size, 1)) # Haven't use yet 

        D_params = lasagne.layers.get_all_params(self.D,trainable=True)
        G_params = self.autoencoder.G_params
        
        real_out = lasagne.layers.get_output(self.D, inputs= self.real_pair, trainable=True)
        # Create expression for passing fake data through the discriminator
        fake_out = lasagne.layers.get_output(self.D, inputs=self.generator_pair, trainable=True)
            
        ##G loss
        ##lam
        lam = 1e-1

        '''
            for vanilla GAN loss
        '''

        '''
        generator_loss_1 = lasagne.objectives.binary_crossentropy(fake_out, 1).mean()
                        
        generator_loss = generator_loss_1 + lam*combine_obj

        discriminator_loss = (lasagne.objectives.binary_crossentropy(real_out, self.true_label)
            + lasagne.objectives.binary_crossentropy(fake_out, self.false_label)).mean()
        '''

        
        # for wGAN
        ## minimize(-fake)
        generator_loss_1 = fake_out.mean()
        generator_loss = generator_loss_1 + lam*combine_obj
        ## maximize (real-fake) == minimize(fake-real)
        discriminator_loss = (real_out.mean() - fake_out.mean())
        


        ## test for loss
        #loss = generator_loss + discriminator_loss 
        eta = theano.shared(lasagne.utils.floatX(2e-5))
        ##generator
        
        updates = lasagne.updates.rmsprop(
            -generator_loss, G_params, learning_rate=eta)
        updates2 = lasagne.updates.rmsprop(
           -discriminator_loss, D_params, learning_rate=eta)
        
        # for vanilla GAN
        
        #updates = lasagne.updates.adam(
        #   generator_loss, G_params, learning_rate=eta, beta1=0.5)
        """
        lasagne.updates.adam will return a dictionary updates the dict could add operation
        """
       

        #updates.update(lasagne.updates.adam(
        #    discriminator_loss, D_params, learning_rate=eta, beta1=0.5))


        # for vanilla GAN 
        
        #updates2 = lasagne.updates.adam(
        #    discriminator_loss, D_params, learning_rate=eta, beta1=0.5)
       
        # Clip critic parameters in a limited range around zero (except biases)
        for param in lasagne.layers.get_all_params(self.D, trainable=True,regularizable=True):
            updates2[param] = T.clip(updates2[param],-self.clip, self.clip)


        updates3 = lasagne.updates.adam(
            combine_obj, G_params, learning_rate=eta, beta1=0.5)

 #print("updates after",updates)
        '''
        self.train_fn = theano.function(inputs=[self.input_var1, self.input_var2],
                                outputs=[(real_out > .5).mean(),
                                (fake_out < .5).mean()],
                                                 updates=updates)
        '''
        self.temp_train = theano.function(inputs=[self.input_var1,self.output_var],
                                          outputs=[combine_obj],
                                          updates=updates3,
                                          allow_input_downcast=True)

        self.G_train_fn = theano.function(inputs=[self.input_var1, self.input_var2, self.output_var],
                                outputs=[generator_loss_1, real_out.mean(), fake_out.mean(), generator_loss-generator_loss_1],
                                 updates=updates)
        '''
        self.D_train_fn = theano.function(inputs=[self.input_var1, self.input_var2, self.true_label, self.false_label],
                                outputs=discriminator_loss,
                                 updates=updates2,allow_input_downcast=True)
        '''
        # for wGAN
        self.D_train_fn = theano.function(inputs=[self.input_var1, self.input_var2],
                                outputs=discriminator_loss,
                                 updates=updates2,allow_input_downcast=True)

    def generateNoise(self, pi=0.25):
        self.pi = pi
        pi_array = np.random.rand(self.batch_size,1)
        #true_label_noise = np.zeros((self.batch_size,1))
        #false_label_noise = np.zeros((self.batch_size,1))

        ### for smoothing
        # 1 -> [0.8,1.2]
        # 0 -> [0, 0.2]
        true_label_noise = np.random.rand(self.batch_size,1)*0.4 + 0.8
        false_label_noise = np.random.rand(self.batch_size,1)*0.2

        ### for noise
        # true -> false [0, 0.2]
        # false -> true [0.8, 1.2]
        true_label_noise[pi_array <= self.pi] = random.random()*0.2
        false_label_noise[pi_array <= self.pi ] = random.random()*0.4+0.8

        return true_label_noise, false_label_noise
        # if random.random() > self.pi:
            # self.true_label_noise = 1
            # self.false_label_noise = 0
        # else:
            # self.true_label_noise = 0
            # self.false_label_noise = 1

'''
if __name__ == "__main__":
    # loading decoder weights and encoder weight
    load_weights(model.net, path='scripts/gen_', epochtoload=90)
    load_weights_test(model.net['output_encoder'], path='fix_3layers_weights/auto_finetune_', epochtoload=60)
'''
