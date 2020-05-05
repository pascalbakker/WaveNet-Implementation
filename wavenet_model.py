import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Add, Activation, Multiply, Dense, Flatten, Input
from tensorflow.keras import Model

"""
This file contains t
"""

class BlockLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, filter_size, dilation_rate):
        super(BlockLayer, self).__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dilation_rate = dilation_rate
        self.tanh_ = Conv1D(filters=self.num_filters, kernel_size=self.filter_size, dilation_rate=self.dilation_rate,
                            padding='same', activation='tanh')
        self.sigmoid_ = Conv1D(filters=self.num_filters, kernel_size=self.filter_size
                               , dilation_rate=self.dilation_rate, padding='same', activation='sigmoid')
        self.skipout_ = Conv1D(1, 1, activation='relu', padding="same")

    def call(self, input_):
        residual = input_
        tanh_out = self.tanh_(input_)
        sigmoid_out = self.sigmoid_(input_)
        combine = Multiply()([tanh_out, sigmoid_out])
        skip_out = self.skipout_(combine)
        output = Add()([skip_out, residual])
        return output, skip_out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'filter_size': self.filter_size,
            'dilation_rate': self.dilation_rate,
        })
        return config

class WaveNet(Model):
    def __init__(self, num_filters, filter_size, dilation_rate, num_layers, input_size):
        super(WaveNet, self).__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.input_size = input_size
        self.finalDense = Dense(self.input_size, activation='softmax')

    def call(self, x, **kwargs):

        # Build initial block
        A, B = BlockLayer(self.num_filters, self.filter_size, self.dilation_rate)(x)
        skip_connections = [B]
        # Build Connections for each layer
        for i in range(self.num_layers):
            dilation_rate = 2 ** ((i + self.dilation_rate) % 9)
            A, B = BlockLayer(self.num_filters, self.filter_size, dilation_rate)(A)
            skip_connections.append(B)

        # Combine final layer
        main_network = Add()(skip_connections)
        main_network = Activation('relu')(main_network)
        main_network = Conv1D(1, 1, activation='relu')(main_network)
        main_network = Conv1D(1, 1)(main_network)
        main_network = Flatten()(main_network)
        return self.finalDense(main_network)

    # call this method to build a NN with just adio
    def model(self):
        x = Input(shape=(self.input_size, 1))
        return Model(x, self.call(x))
