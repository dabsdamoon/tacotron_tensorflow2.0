import tensorflow as tf


##### Class to get prenet

class Prenet(tf.keras.layers.Layer):
    
    """
    Class to get Prenet (just two dense layers).
    """

    def __init__(self, units, dropout_rate, **kwargs):
        
        """
        Args:
            units: Number of units in dense layer. The second layer will contain half number of units compared to the first layer.
            dropout_rate: Dropout rate for Dropout layer.
            **kwargs : Dictionary that contains other common argument for layer creation.        
        """

        super(Prenet, self).__init__(**kwargs)        
        
        self.W1 = tf.keras.layers.Dense(units)        
        self.W2 = tf.keras.layers.Dense(units)
                                        
        self.LR = tf.keras.layers.LeakyReLU(0.2)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs):
        
        """
        Args:
            inputs: Input tensor of the first dense layer.

        Returns:
            outputs: Output of the seconc layer
        """

        l1 = self.dropout(self.LR(self.W1(inputs)))
        outputs = self.dropout(self.LR(self.W2(l1)))

        return outputs


##### Function to get batch normalization

def bn(inputs):

    """
    Just converted Kyubyoung's bn function from tensorflow v1.0 to tensorflow v2.0

    Args:
        inputs: Input tensor.

    Returns:
        outputs: Output tensor with batch normalization. Notice that the function applies fused = True when rank > 1.
    """

    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.expand_dims(inputs, axis=2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis=1)

        outputs = tf.keras.layers.BatchNormalization(center=True,
                                                     scale=True,
                                                     fused=True)(inputs)       # restore original shape
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis=[1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis=1)
    else:
        outputs = tf.keras.layers.BatchNormalization(center=True,
                                                     scale=True,
                                                     fused=False)(inputs)

    return outputs


##### Function to get convolutional banks

def conv1d_banks(inputs, hp, K=16, use_bias = False, naming = None):

    """
    Function to construct conv1d bank architecture.

    Args:
        inputs: Input tensor.
        hp: Class of hyperparameters. Conv1D will have filter size of embedding_dimension // 2
        K: Size of filters. Default value is 16.
        use_bias: Whether to use bias for convolution layer. Default is false.
        naming: Format of name (scope) in each layers of conv1d_banks function.

    Returns:
        outputs: Output tensor with shape (Batch_size, Timestep, embedding_dimension // 2 * K)
    """

    outputs = tf.keras.layers.Conv1D(hp.embed_size//2, 
                                     1, 
                                     padding = "same", 
                                     use_bias = use_bias,
                                     name = "{}_conv1d_banks_{}".format(naming, 1))(inputs)

    for k in range(2, K + 1):
        output = tf.keras.layers.Conv1D(hp.embed_size//2, 
                                        k, 
                                        padding = "same", 
                                        use_bias = use_bias, 
                                        name = "{}_conv1d_banks_{}".format(naming, k))(inputs)
        
        outputs = tf.concat((outputs, output), -1)
        
    outputs =tf.keras.layers.BatchNormalization(center=True,
                                                scale=True,
                                                fused=False,
                                                name = "{}_BN".format(naming))(outputs)
    
    outputs = tf.keras.layers.LeakyReLU(0.2)(outputs)
    
    return outputs


##### Function to get highwaynet

def highwaynet(inputs, num_units=None, naming = None):

    """
    Function to construct highway network architecture.
    For more information, read: https://arxiv.org/pdf/1505.00387.pdf

    Args:
        inputs: Input tensor.
        num_units: Number of units for dense layer used in highway network.
        naming: Format of name (scope) in each layer of highwaynet

    Returns:
        outputs: Output tensor

    """

    if not num_units:
        num_units = inputs.get_shape()[-1]
        
    H = tf.keras.layers.Dense(num_units, 
                              activation= "relu", 
                              name = "{}_highway_H".format(naming))(inputs)
    
    T = tf.keras.layers.Dense(num_units, 
                              activation= "sigmoid", 
                              name = "{}_highway_T".format(naming),
                              bias_initializer=tf.constant_initializer(-1.0))(inputs)
    
    outputs = H*T + inputs*(1.0 - T)
    
    return outputs

