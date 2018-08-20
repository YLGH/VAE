#!/usr/bin/env python


# In[2]:


import decoders
import encoders
import latent
from VAE_base import VAE_base

import numpy as np


# In[3]:


from get_json import get_easy
data_attributes, data_attributes_attack = get_easy()


# In[4]:


class LSTM_Gaussian_LSTM(VAE_base):
    def __init__(self, encoder_dim=512, latent_dim=64, decoder_dim=512, *args, **kwargs):
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim
        super(LSTM_Gaussian_LSTM, self).__init__(*args, **kwargs)

    def define_encoder(self):
        encoders.encoder_LSTM(self, self.encoder_dim)
          
    def define_latent(self):
        latent.latent_gaussian(self, self.latent_dim)
        
    def define_decoder(self):
        decoders.decoder_LSTM(self, self.decoder_dim)

class LSTM_Gaussian_DilatedConv(VAE_base):
    def __init__(self, 
                 encoder_dim=512, 
                 latent_dim=64, 
                 decoder_dim=512, 
                 kernel_sizes=[], 
                 dilation_rates=[],
                 *args, **kwargs):
        super(LSTM_Gaussian_DilatedConv, self).__init__(*args, **kwargs)

        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        
        print('Encoder Dim ', encoder_dim)
        print('Latent Dim ', latent_dim)
        print('Decoder Dim ', decoder_dim)
        print('Kernel Sizes', kernel_sizes)
        print('Dilation Rates', dilation_rates)

        self.hash_list.append(self.encoder_dim)
        self.hash_list.append(self.latent_dim)
        self.hash_list.append(self.decoder_dim)
        self.hash_list.append(str(self.kernel_sizes))
        self.hash_list.append(str(self.dilation_rates))

    def define_encoder(self):
        encoders.encoder_LSTM(self, self.encoder_dim)
          
    def define_latent(self):
        latent.latent_gaussian(self, self.latent_dim)
        
    def define_decoder(self):
        decoders.decoder_dilated_convolution(self, 
                                             self.decoder_dim, 
                                             self.kernel_sizes, 
                                             self.dilation_rates)


# In[5]:


data_attributes_all = data_attributes['all']


# In[6]:


vae = LSTM_Gaussian_DilatedConv(latent_dim=512,
                                decoder_dim=784,
                                train_data = data_attributes_all, 
                                model_name = 'model1_dilate_all',
                                kl_beta = 0.4,
                                max_sequence_length=198,
                                kernel_sizes=[1,3,3,3,3,3,3,3,1], 
                                dilation_rates=[1,1,2,4,8,16,32,64,1],)
vae.define_model()


# In[7]:


vae.load_weights('ckpt', load_latest=True)


def get_embeddings(attributes, batch_size=512):
    embeddings = None
    start = 0
    while start < len(attributes):
        if embeddings is None:
            embeddings = vae.sentence_to_params(attributes[start:min(start+batch_size, len(attributes))])[0]
        else:
            embeddings = np.vstack([embeddings, vae.sentence_to_params(attributes[start:min(start+batch_size, len(attributes))])[0]])
        start += batch_size
    return embeddings

reg_embeddings = get_embeddings(data_attributes['all'], 128)
att_embeddings = get_embeddings(data_attributes_attack['all'], 128)


# In[65]:

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam


classifier = Sequential([
    Dense(128, activation='relu', input_shape=(vae.latent_dim,)),
    Dropout(0.25),
    Dense(1, activation='sigmoid'),
])

classifier.compile(optimizer=Adam(3e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

classifier.fit(x=np.vstack([reg_embeddings, att_embeddings]),
               y=np.concatenate([np.zeros(len(reg_embeddings)), np.ones(len(att_embeddings))]), 
               batch_size=64, epochs=50, verbose=1, validation_split=0.2,
               shuffle=True)

