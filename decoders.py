from keras.layers import Input, LSTM, RepeatVector, Concatenate, TimeDistributed, Dense, BatchNormalization, Conv1D, Dropout
import keras


def decoder_LSTM(self, lstm_size):
    self.decoder_input = Input(
        shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32', name='decoder_input')

    decoder_inputs = RepeatVector(
        self.MAX_SEQUENCE_LENGTH)(self.z_h)

    embedded_decoder_input = self.embedding_layer(
        self.decoder_input)

    decoder_inputs = Concatenate(
        axis=-1)([decoder_inputs, embedded_decoder_input])

    decoder_lstm = LSTM(lstm_size,
                        return_sequences=True,
                        return_state=True,
                        dropout=0.5,
                        name='decoder_lstm')
    
    self.decoder_name='LSTM'

    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs)
    
    self.decoder_distribution_outputs = TimeDistributed(Dense(len(self.tokenizer.word_index),
                                                              activation='softmax',
                                                            ))(decoder_inputs)


def decoder_dilated_convolution(self, filters, kernel_sizes=[], dilation_rates=[]):
    
    assert len(kernel_sizes) == len(dilation_rates)
    
    self.decoder_input = Input(
        shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32', name='decoder_input')

    decoder_inputs = RepeatVector(
        self.MAX_SEQUENCE_LENGTH)(self.z_h)
    embedded_decoder_input = self.embedding_layer(
        self.decoder_input)

    decoder_inputs = Concatenate(
        axis=-1)([decoder_inputs, embedded_decoder_input])
    
    for kernel_size, dilation_rate in zip(kernel_sizes, dilation_rates):
        self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=filters,
                                                                     kernel_size=kernel_size,
                                                                     dilation_rate=dilation_rate,
                                                                     activation='relu',
                                                                     padding='causal')(decoder_inputs)))
    self.decoder_conv = Concatenate(
        axis=-1)([decoder_inputs, self.decoder_conv])

    self.decoder_distribution_outputs = TimeDistributed(Dense(len(self.tokenizer.word_index),
                                                              activation='softmax'))(self.decoder_conv)

    self.decoder_name = 'dilated_conv_' + str(list(zip(kernel_sizes, dilation_rates)))

