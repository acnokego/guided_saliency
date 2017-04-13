import lasagne
import theano
import theano.tensor as T
import numpy as np
import encoder
import decoder
import style_transfer as style

def build_autoencoder(encoder, input_height, input_width):
     autoencoder = decoder.build(encoder, input_height, input_width)
     return autoencoder

class autoencoder(object):
    def __init__(self, input_width, input_height, lr= 3e-4, c_lr=1e-3, batch_size=8):

        self.inputWidth = input_width
        self.inputHeight = input_height
        self.input_var = T.tensor4()
        self.output_var = T.tensor4()
        self.batch_size = batch_size
        self.input_img = T.tensor4()

        ##self.net
        #self.decoder = decoder.build(self.inputWidth, self.inputHeight, self.input_var)
        self.encoder = encoder.build(self.inputHeight, self.inputWidth, self.input_var)
        self.net = build_autoencoder(self.encoder, self.inputHeight, self.inputWidth)
        self.style_net = style.build_model(self.inputHeight, self.inputWidth, self.input_img)
        style.load_weights(self.style_net)

        self.net = style.connect(self.net)
        style.load_weights(self.net)
        
        
        ### For autoencoder
        output_layer_name = 'output'
        generated_layer_name = 'output_encoder'
        prediction = lasagne.layers.get_output(self.net[output_layer_name])

        test_prediction = lasagne.layers.get_output(self.net[generated_layer_name], 
                                                    deterministic=True)
        self.predictFunction = theano.function([self.input_var], test_prediction)

        output_var_pooled = T.signal.pool.pool_2d(self.output_var, (4, 4), 
                                mode="average_exc_pad", ignore_border=True)
        prediction_pooled = T.signal.pool.pool_2d(prediction, (4, 4), 
                                mode="average_exc_pad", ignore_border=True)
        
        ##objective L2 loss
        obj = lasagne.objectives.binary_crossentropy(prediction_pooled, 
            output_var_pooled).mean() + lasagne.regularization.regularize_network_params(
            self.net[generated_layer_name],lasagne.regularization.l2)

        train_err = obj

        G_params = lasagne.layers.get_all_params(self.net[generated_layer_name], trainable=True)
        self.G_lr = theano.shared(np.array(lr, dtype=theano.config.floatX))
        G_updates = lasagne.updates.nesterov_momentum(train_err, G_params, 
                                    learning_rate=self.G_lr, momentum=0.5)

        self.G_trainFunction = theano.function(inputs=[self.input_var, self.output_var]
                            , outputs=train_err, updates= G_updates, allow_input_downcast=True)

        # content feature extraction
        #feature_layers = ['conv4_2_s', 'conv1_1_s', 'conv2_1_s', 'conv3_1_s', 'conv4_1_s', 'conv5_1_s']
        feature_layers = 'conv4_2_s'
        #input_feature_layers = {k:self.style_net[k] for k in feature_layers}
        #generated_feature_layers = {k:self.net[k] for k in feature_layers}
        
        outputs = lasagne.layers.get_output(self.style_net[feature_layers], 
                                            inputs=self.input_img)
        generated_outputs = lasagne.layers.get_output(self.net[feature_layers],
                                            inputs=self.input_var)
        
       # print type(outputs)
       # print outputs
        #image_feat = theano.shared(np.array(outputs, dtype=theano.config.floatX))
        #generated_feat = theano.shared(np.array(generated_outputs, dtype=theano.config.floatX))
        #generated_feat = theano.function([self.input_var], generated_outputs)
        #image_feat = theano.function([self.input_img], outputs)

        G_params_2 = lasagne.layers.get_all_params(self.net[generated_layer_name], trainable=True)
        ## content loss
 #       for i in range(len(feature_layers)):

        obj2 = lasagne.objectives.squared_error(generated_outputs, outputs).sum()
      #  + lasagne.regularization.regularize_network_params(self.net['output_encoder'],
      #  lasagne.regularization.l2)
        train_err2 = obj2
        self.c_lr = theano.shared(np.array(c_lr, dtype=theano.config.floatX))
        content_update = lasagne.updates.adam(train_err2, G_params_2,learning_rate=self.c_lr,
                                              beta1=0.9, beta2=0.999, epsilon=1e-8)


        self.content_trainFunction = theano.function(inputs=[self.input_var, self.input_img],
                                     outputs=train_err2, updates=content_update, 
                                     allow_input_downcast=True)
    
        
        




