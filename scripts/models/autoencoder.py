import lasagne
import theano
import theano.tensor as T
import numpy as np
import encoder
import decoder
import style_transfer as style
import vgg16
from constants import INPUT_SIZE


def build_autoencoder(encoder, input_height, input_width):
     autoencoder = decoder.build(encoder, input_height, input_width)
     return autoencoder

class autoencoder(object):
    def __init__(self, input_width, input_height, lr= 1e-5, c_lr=1e-5, lam=1e-9, batch_size=16):

        self.inputWidth = input_width
        self.inputHeight = input_height
        self.input_var = T.tensor4()
        self.output_var = T.tensor4()
        self.batch_size = batch_size
        self.input_img = T.tensor4()
        print('building autoencoder.......')
        ##self.net
        #self.decoder = decoder.build(self.inputWidth, self.inputHeight, self.input_var)
        self.encoder = encoder.build(self.inputHeight, self.inputWidth, self.input_var)
        self.decoder = decoder.build(None, self.inputHeight, self.inputWidth)

        #self.net = self.encoder
        '''
        #self.net = build_autoencoder(self.encoder, INPUT_SIZE[1], INPUT_SIZE[0])
        #self.vgg_net = vgg16.build(None, self.inputHeight, self.inputWidth, False, self.input_img)
        #decoder.set_pretrained_weights(self.vgg_net)

        #self.style_net = style.build_model(self.inputHeight, self.inputWidth, self.input_img)
        #style.load_weights(self.style_net)

        #self.net = style.connect(self.net)
        #style.load_weights(self.net)
        '''       
        
        ### For autoencoder
        output_layer_name = 'output'
        generated_layer_name = 'output_encoder'

        fake_out = lasagne.layers.get_output(self.encoder['output_encoder_scaled'], inputs=self.input_var)
        prediction = lasagne.layers.get_output(self.decoder[output_layer_name], inputs=fake_out)

        test_prediction = lasagne.layers.get_output(self.encoder[generated_layer_name], deterministic=True)
        self.predictFunction = theano.function([self.input_var], test_prediction)

        output_var_pooled = T.signal.pool.pool_2d(self.output_var, (2, 2), 
                                mode="average_exc_pad", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (2, 2), 
                                mode="average_exc_pad", ignore_border=True)
        
        ##objective L2 loss
        obj = lasagne.objectives.squared_error(prediction_pooled, 
            output_var_pooled).sum()
        
        #+ lasagne.regularization.regularize_network_params(
        #    self.net[generated_layer_name],lasagne.regularization.l2)

        train_err = obj

        self.G_params = lasagne.layers.get_all_params(self.encoder[generated_layer_name], trainable=True)
        self.G_lr = theano.shared(np.array(lr, dtype=theano.config.floatX))
        #G_updates = lasagne.updates.nesterov_momentum(train_err, G_params,
        '''
        G_updates = lasagne.updates.adam(train_err, G_params,learning_rate=self.G_lr, 
                                              beta1=0.9, beta2=0.999, epsilon=1e-8)

        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var]
                            , outputs=train_err, updates= G_updates, allow_input_downcast=True)
        '''
        # content feature extraction
        #feature_layers = ['conv4_2_s']
        
        feature_layers = ['conv4_2_de', 'conv3_1_de', 'conv2_1_de']
   #     feature_layers_2 = 'conv5_2_s'
        input_feature_layers = {k:self.decoder[k] for k in feature_layers}
        generated_feature_layers = {k:self.decoder[k] for k in feature_layers}
        
        outputs = lasagne.layers.get_output(input_feature_layers.values(), 
                                            inputs=self.input_img)
        generated_outputs = lasagne.layers.get_output(generated_feature_layers.values(),
                                            inputs=fake_out)

        ## content loss
        w = float(1.0/len(feature_layers))
        obj2 = 0
        print (w)
        for i in range(len(feature_layers)):
            obj2 = obj2 + w*style.content_loss(generated_outputs[i], outputs[i])


        

        train_err2 = obj2

        #combine_obj = train_err + lam*train_err2
        self.combine_obj = train_err + lam*train_err2
        
        G_updates = lasagne.updates.adam(self.combine_obj, self.G_params,learning_rate=self.G_lr, 
                                              beta1=0.9, beta2=0.999, epsilon=1e-8)
    
        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var, self.input_img]
                            , outputs=self.combine_obj, updates= G_updates, allow_input_downcast=True)
        
        '''
        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var]
                            , outputs=self.combine_obj, updates= G_updates, allow_input_downcast=True)
        '''
        '''
        self.G_trainFunction = theano.function(inputs=[self.input_var, self.input_img]
                            , outputs=self.combine_obj, updates= G_updates, allow_input_downcast=True)
        '''
        '''
        self.c_lr = theano.shared(np.array(c_lr, dtype=theano.config.floatX))
        content_update = lasagne.updates.adam(train_err2, G_params,learning_rate=self.c_lr,
                                              beta1=0.9, beta2=0.999, epsilon=1e-8)


        self.content_trainFunction = theano.function(inputs=[self.input_var, self.input_img],
                                     outputs=train_err2, updates=content_update, 
                                     allow_input_downcast=True)
    
        '''       
        




