from keras.layers import Dense, Lambda, Reshape
from keras import backend as K
from keras.activations import softmax
from keras.models import Model


def latent_gaussian(self, latent_dim):
    self.z_h_mean_layer = Dense(latent_dim,
                                name='z_h_mean',
                                activation='linear')

    self.z_h_log_var_layer = Dense(latent_dim,
                                   name='z_h_log_var',
                                   activation='linear')

    self.z_h_mean = self.z_h_mean_layer(
        self.h)
    self.z_h_log_var = self.z_h_log_var_layer(
        self.h)

    def sampling_batch(args):
        z_mean, z_log_var = args
        batch_size = K.shape(z_mean)[0]
        epsilon = K.random_normal(
            shape=(batch_size, latent_dim), mean=0.0, stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    self.z_h = Lambda(sampling_batch, output_shape=(
        latent_dim,), name='z_h')([self.z_h_mean, self.z_h_log_var])

    self.kl_loss = - 0.5 * \
        K.mean(1 + self.z_h_log_var - K.square(self.z_h_mean) -
               K.exp(self.z_h_log_var))

    self.latent_name = '{}_{}'.format(
        'gaussian', latent_dim)

    self.latent_type = 'gaussian'
    
    self.get_embedding = self.z_h_mean_layer


def latent_gumbel(self, M, N):
    self.M = M
    self.N = N

    self.tau = K.variable(
        5.0, name="temperature")

    self.logits_layer = Dense(
        self.M * self.N, activation='relu', name='logits_layer')

    self.reshape_layer = Reshape((-1, self.M,self.N))

    self.logits = self.logits_layer(self.h)
    self.logits = self.reshape_layer(self.logits)
    self.logits = softmax(self.logits)

    def sampling(logits_y):
        U = K.random_uniform(
            K.shape(logits_y), 0, 1)
        # logits + gumbel noise
        y = logits_y - \
            K.log(-K.log(U + 1e-20) + 1e-20)
        y = softmax(y/self.tau)
        y = K.reshape(
            y, (-1, self.M * self.N))
        return y

    self.z_h = Lambda(
        sampling, name='z_h')(self.logits)

    self.kl_loss = -K.sum(self.logits*K.log(1e-20 + self.N*self.logits), axis=(-1,-2))

    self.latent_name = '{}_{}_{}'.format(
        'gumbel', self.M, self.N)
    self.latent_type = 'gumbel'

    def get_embedding(h):
        e = self.logits_layer(h)
        e = self.reshape_layer(e)
        return softmax(e)

    self.get_embedding = Lambda(get_embedding, name='embedding')

