from abc import ABC, abstractmethod
from VAE_base_keras_imports import *

import tensorflow as tf

class VAE_base(ABC):
    def __init__(self,
                 train_data,
                 model_name,
                 kl_beta=1.0,
                 embedding_dim=16,
                 max_sequence_length=30,
                 max_num_words=200):
        
        path = Path(
            '{}/'.format(model_name))
        if not path.exists():
            path.mkdir()

        self.MAX_SEQUENCE_LENGTH = max_sequence_length
        self.EMBEDDING_DIM = embedding_dim

        self.kl_beta = kl_beta
        self.KL_error = K.variable(0.0)
        self.model_name = model_name

        self.train_data = train_data

        self.tokenizer = Tokenizer(num_words=max_num_words,
                                   filters='',
                                   lower=False, 
                                   char_level=True, 
                                   oov_token='ô')
        
        self.tokenizer.fit_on_texts(self.train_data)
        self.tokenizer.word_index['</e>'] = len(self.tokenizer.word_index)+1
        self.tokenizer.word_index['</s>'] = len(self.tokenizer.word_index)+1
        
        self.index_word = {}
        
        for word in self.tokenizer.word_index:
            self.index_word[self.tokenizer.word_index[word]] = word
        
        self.embedding_layer = Embedding(input_dim=len(self.tokenizer.word_index),
                                         output_dim=self.EMBEDDING_DIM,
                                         input_length=self.MAX_SEQUENCE_LENGTH,
                                         trainable=True)

        print('Found %s unique tokens.' %
              len(self.tokenizer.word_index))

#   Sets hidden state that is passed onto latent layer
    @abstractmethod
    def define_encoder(self):
        pass

#   Samples latent z that is passed onto decoder.
    @abstractmethod
    def define_latent(self):
        pass

#   Sets decoder outputs
    @abstractmethod
    def define_decoder(self):
        pass

    def define_model(self):
        self.define_encoder()
        self.define_latent()
        self.define_decoder()

        def vae_loss(x, x_cat):
            reconstruction_loss = K.sum(objectives.categorical_crossentropy(x, x_cat), axis=-1)
            loss = reconstruction_loss + self.KL_error*self.kl_loss

            return loss

        self.model_vae = Model(
            [self.sequence_input, self.decoder_input], self.decoder_distribution_outputs)
        self.model_vae.compile(optimizer=keras.optimizers.adam(
            lr=1e-3, beta_1=0.5), loss=[vae_loss])

        
        self.model_vae_params = Model(
            self.sequence_input, self.z_h_mean)
        self.get_params = K.function([self.sequence_input], [
            self.z_h_mean, self.z_h_log_var])
        self.generate = K.function([self.z_h, self.decoder_input], [
                                   self.decoder_distribution_outputs])

    def generator_text_in(self, batch_size):
        while True:
            input_text_data = random.sample(
                self.train_data, batch_size)
            batch_input_text_data = self.tokenizer.texts_to_sequences(
                input_text_data)
            batch_input_text_data = pad_sequences(
                batch_input_text_data, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

            batch_decoder_inputs = np.roll(
                batch_input_text_data, 1)
            batch_decoder_inputs[:,0] = self.tokenizer.word_index['</s>']

            batch_decoder_sequence_data = np.zeros(
                (batch_size, self.MAX_SEQUENCE_LENGTH, len(self.tokenizer.word_index)))

            for i in range(batch_size):
                for token_index, word_index in enumerate(
                        batch_input_text_data[i]):
                    word_index = int(
                        word_index)
                    batch_decoder_sequence_data[i][token_index][word_index] = 1.0

            yield [batch_input_text_data, batch_decoder_inputs], batch_decoder_sequence_data

    def load_weights(self, save_name):
        save_path = Path('{}/{}'.format(self.model_name, save_name))
        self.model_vae.load_weights(str(save_path))

    def train(self,
              batch_size=512,
              start_epoch=0,
              end_epoch=10,
              save_every=10,
              steps_per_epoch=100,
              save_name='ckpt'):

        print('Training ', '{}/{}-{}-{}'.format(self.model_name,
                                                self.encoder_name,
                                                self.latent_name,
                                                self.decoder_name))
        

        print([self.train_data[0], self.train_data[-1]])

        kl_value = 0.0
        for epoch in range(start_epoch, end_epoch):
            print('EPOCH : ', epoch)
            kl_value = min(self.kl_beta, (4/3)*epoch/end_epoch)
            
            K.set_value(
                self.KL_error, kl_value)

            self.model_vae.fit_generator(self.generator_text_in(batch_size),
                                         steps_per_epoch=steps_per_epoch,
                                         epochs=1)
            
            self.decode_sequence([self.train_data[0], self.train_data[-1]], False)

            

            if (epoch + 1) % save_every == 0:
                save_path = Path('{}/{}'.format(self.model_name, save_name+str(epoch+1)))
                self.model_vae.save_weights(str(save_path))


    def generate_from_mu(self, mu):
        decoder_in = np.ones(
            (self.MAX_SEQUENCE_LENGTH), dtype=int)
        next_index = self.tokenizer.word_index['</s>']
        decoded_sequence = ""
        for index in range(self.MAX_SEQUENCE_LENGTH):
            decoder_in[index] = next_index
            predict = self.generate(
                [[mu], [decoder_in]])[0][0][index]
            most_likely_word_index = np.argmax(
                predict)
            if most_likely_word_index in self.index_word:
                decoded_sequence += self.index_word[most_likely_word_index] + " "
        return decoded_sequence

    def sentence_to_params(self, sequence):
        batch_input_text_data = self.tokenizer.texts_to_sequences(sequence)
        batch_input_text_data = pad_sequences(batch_input_text_data, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')
        return np.asarray(self.get_params([batch_input_text_data]))

    def decode_sequence(self, sequences, print_original=True):
        batch_input_text_data = self.tokenizer.texts_to_sequences(
            sequences)
        batch_input_text_data = pad_sequences(
            batch_input_text_data, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        batch_decoder_inputs = np.roll(
            batch_input_text_data, 1)
        batch_decoder_inputs[:,0] = self.tokenizer.word_index['</s>']
        params = self.model_vae.predict([batch_input_text_data, batch_decoder_inputs])

        for sequence_index, sequence in enumerate(params):
            if print_original:
                print('Actual Sequence : ',
                      sequences[sequence_index])
            decoded_sequence = ""
            for word_in_sequence_index in sequence:
                most_likely_word_index = np.argmax(word_in_sequence_index)
                if most_likely_word_index in self.index_word:
                    decoded_sequence += self.index_word[most_likely_word_index]
            print("Decoded Sentence :", decoded_sequence)
            print("================================")
