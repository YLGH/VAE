from keras.layers import Input, LSTM, Bidirectional, Concatenate, Lambda
from keras import backend as K

def encoder_LSTM(self, lstm_size):
    self.sequence_input = Input(
        shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32', name='encoder_input')

    self.embedded_input_sequences = self.embedding_layer(
        self.sequence_input)

    self.encoder_layer = LSTM(lstm_size,
                              return_sequences=True,
                              return_state=True,
                              dropout=0.5)
    

    _, self.h, _ = self.encoder_layer(
        self.embedded_input_sequences)

    self.encoder_name = '{}_{}'.format(
        'LSTM', lstm_size)

    def get_h(sequence_embedding):
        _, h, _ = self.encoder_layer(
            sequence_embedding)
        return h
    
    self.get_h = get_h


def encoder_BiLSTM(self, lstm_size):

    self.sequence_input = Input(
        shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32', name='encoder_input')
    self.embedded_input_sequences = self.embedding_layer(
        self.sequence_input)

    self.encoder_layer = Bidirectional(LSTM(lstm_size,
                                            return_sequences=True,
                                            return_state=True,
                                            dropout=0.5))

    _, self.h_1, _, self.h_2, _ = self.encoder_layer(
        self.embedded_input_sequences)
    self.h = Concatenate(
        axis=-1)([self.h_1, self.h_2])

    self.encoder_name = '{}_{}'.format(
        'BiLSTM', lstm_size)

    def get_h(sequence_embedding):
        _, h_1, _, h_2, _ = self.encoder_layer(
            sequence_embedding)
        h = Concatenate(
            axis=-1)([h_1, h_2])
        return h

    self.get_h = get_h


