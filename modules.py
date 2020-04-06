##### Import installed packages
import tensorflow as tf

##### Import custom packated
from networks import *
from attention import *


##### Class to get embedding with mask

class EmbeddingwithMask(tf.keras.layers.Layer):
    """
    Customized layer to return embedding and mask at the same time.
    """
    def __init__(self, hparams, **kwargs):
        
        """
        Args:
            units: Number of units in dense layers in attention.
            normalize: Whether to normalize the score. If true, additional weights are used to compute normalized scores.
            **kwargs : Dictionary that contains other common argument for layer creation.        
        """

        super(EmbeddingwithMask, self).__init__(**kwargs)

        self.hp = hparams
        self.embedding = tf.keras.layers.Embedding(len(self.hp.vocab), 
                                                   self.hp.embed_size, 
                                                   embeddings_initializer = tf.keras.initializers.TruncatedNormal(mean = 0.0, stddev=0.01),
                                                   mask_zero = True)


    
    def call(self, inputs):

        """
        Args:
            inputs: Input tensor.

        Returns:
            embedded_inputs: Input tensor embedded.
            embedded_mask: Mask when computing embedding. Consider 0 as mask value.
        """

        embedded_inputs = self.embedding(inputs)
        embedded_mask = self.embedding.compute_mask(inputs)

        return embedded_inputs, embedded_mask


##### Function to get encoder

def get_encoder(hp):
    
    """
    Args:
        hp: Class of hyperparameters. See hyperparams.py.

    Returns:
        model: tf.keras.Model object with taking inputs as input layer and [memory, final_state, mask] as return tensors.
    """

    input_shape = (None, )
    inputs = tf.keras.layers.Input(shape = input_shape)
    
    ##### Embedding and prenet
    embedded_input, mask =  EmbeddingwithMask(hp)(inputs)
    prenet_out = Prenet(hp.encoder_prenet_size, hp.dropout_rate)(embedded_input)

    # print("prenet_out: {}".format(prenet_out.shape))

    ##### Convolutional banks    
    enc = conv1d_banks(prenet_out, hp, K=hp.encoder_num_banks, naming = "encoder")
    enc = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="same", name = "encoder_maxpooling")(enc)

    ##### Convolution layer
    enc = tf.keras.layers.Conv1D(filters=hp.encoder_conv_channels,
                                 kernel_size=hp.encoder_kernel_size, 
                                 padding = "same", 
                                 use_bias = False, 
                                 name = "encoder_cov1d_1")(enc)

    enc = bn(enc)
    enc = tf.keras.layers.LeakyReLU(0.2)(enc)

    enc = tf.keras.layers.Conv1D(filters=hp.encoder_conv_channels,
                                  kernel_size=hp.encoder_kernel_size, 
                                  padding = "same", 
                                  use_bias = False, 
                                  name = "encoder_conv1d_2")(enc)

    enc = bn(enc)

    # print("enc: {}".format(enc.shape))


    ##### Residual connection
    enc += prenet_out
    
    ##### Highway networks
    for i in range(hp.num_highwaynet_blocks):
        enc = highwaynet(enc, num_units=hp.encoder_conv_channels, naming = "encoder_{}".format(i)) # Highway nets; (N, T_x, E)

    ##### Bidirectional GRU
    memory, final_state_f, final_state_b = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hp.embed_size//2,
                                                                                             return_sequences = True, 
                                                                                             return_state = True),
                                                                         name = "encoder_bidirectional")(enc, mask = mask)
    
    final_state = tf.concat([final_state_f, final_state_b], axis = -1)
    
    model = tf.keras.Model(inputs, [memory, final_state, mask])

    return model


##### Function to get Attention Decoder (Decoder1; for mel-spectrogram with reduction factor)

class AttentionDecoder(tf.keras.Model):
    
    def __init__(self, 
                 attention_mechanism, 
                 hparams, 
                 **kwargs):
        
        """
        Args:
            attention_mechanism: Attention mechanism to be used when computing attention. Currently, only Bahdanau Attention mechanism can be applied.
            hparams: Class of hyperparameters. See hyperparams.py.
            **kwargs : Dictionary that contains other common argument for layer creation.        
        """
        
        super(AttentionDecoder, self).__init__(**kwargs)

        self.hp = hparams
        
        self.attention_cell = tf.keras.layers.GRU(self.hp.embed_size,
                                                  return_state = True,
                                                  return_sequences = True)
        
        self.attention_mechanism = attention_mechanism
        self.attention_layer = tf.keras.layers.Dense(self.hp.embed_size) # should be same as the output of prenet
        
        self.prenet = Prenet(self.hp.embed_size, 
                             self.hp.dropout_rate)
        
        self.decoder_cell = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells([tf.keras.layers.GRUCell(self.hp.embed_size // 2),
                                                                                 tf.keras.layers.GRUCell(self.hp.embed_size // 2)]),
                                                return_state = True,
                                                return_sequences = True)
        
        self.projection_layer = tf.keras.layers.Dense(self.hp.n_mels * self.hp.r, 
                                                      name = "decoder1_mel")

        
    def call(self, 
             inputs,
             prev_states,
             **kwargs):

        """
        Args:
            inputs: Input tensors.
            prev_sates: List of tensors from previous timstep used to compute current values.
            **kwargs : Dictionary that contains other common argument for layer creation.        

        Returns:
            final_outputs: Final output tensors.
            [attn_cell_states, decoder_cell_states, alignments, memory, memory_mask]: Tensors going to be used to compute values in next timestep.
        """

        ##### Define previous states
        prev_attn_cell_states, prev_decoder_cell_states, prev_alignments, memory, memory_mask = prev_states
        
        ##### Compute previous attention
        prev_attention, _ = self.attention_mechanism._compute_attention(prev_alignments, memory)
        prev_attention = self.attention_layer(prev_attention)
        prev_attention = tf.expand_dims(prev_attention, axis = 1)

        ##### Process inputs (applying prenet)
        inputs_processed = self.prenet(inputs)
        inputs_processed = tf.expand_dims(inputs_processed, axis = 1)

        ##### Get cell input
        attn_cell_inputs = tf.concat([inputs_processed, prev_attention], axis = -1)
        
        ##### Get cell output and state
        attn_cell_outputs, attn_cell_states  = self.attention_cell(attn_cell_inputs, 
                                                                   prev_attn_cell_states, 
                                                                   **kwargs)        
        
        ##### Compute current attention

        if self.hp.use_monotonic:
            alignments, _ = self.attention_mechanism(tf.squeeze(attn_cell_outputs, axis = 1),
                                                     memory,
                                                     prev_alignments,
                                                     memory_mask)

        else:
            alignments, _ = self.attention_mechanism(tf.squeeze(attn_cell_outputs, axis = 1),
                                                     memory,
                                                     memory_mask)
                
        ##### Compute new attention
        attention, _ = self.attention_mechanism._compute_attention(alignments, memory)

        ##### Concatenate LSTM output and attention text vector
        attn_outputs_concat = tf.concat([tf.squeeze(attn_cell_outputs, axis = 1), 
                                         attention], 
                                         axis = -1)
    
        ##### Go through stacked decoder RNN layer
        
        decoder_cell_outputs, ds1, ds2 = self.decoder_cell(tf.expand_dims(attn_outputs_concat, axis = 1),
                                                           prev_decoder_cell_states,
                                                           **kwargs)
        decoder_cell_states = [ds1, ds2]
        
        ##### Get linear projection for the target spectrogram frame
        final_outputs = self.projection_layer(decoder_cell_outputs)

        return final_outputs, [attn_cell_states, decoder_cell_states, alignments, memory, memory_mask]

    
    ##### Function to get the initial alignments
    def _initial_alignments(self, batch_size, alignment_size, dtype):

        """
        Obtained from tensorflow addons github.
        Creates the initial alignment values for the monotonic attentions.
        Initializes to dirac distributions, i.e.
        [1, 0, 0, ...memory length..., 0] for all entries in the batch.
        Args:
          batch_size: `int32` scalar, the batch_size.
          dtype: The `dtype`.
        Returns:
          A `dtype` tensor shaped `[batch_size, alignments_size]`
          (`alignments_size` is the values' `max_time`).
        """

        max_time = alignment_size

        return tf.one_hot(
            tf.zeros((batch_size,), dtype=tf.int32), max_time, dtype=dtype)


##### Function to get decoder2 (for linear spectrogram)

def get_decoder2(hp):

    """
    Args:
        hp: Class of hyperparameters. See hyperparams.py.

    Returns:
        model: tf.keras.Model object with taking inputs as input layer and [memory, final_state, mask] as return tensors.
    """

    input_shape = (None, hp.n_mels * hp.r)
    
    ##### Post CBHG
    mel_hat_input = tf.keras.layers.Input(shape = input_shape)
    
    inputs_processed = tf.keras.layers.Reshape([-1, hp.n_mels])(mel_hat_input)

    dec = conv1d_banks(inputs_processed, hp, K = hp.decoder_num_banks, naming = "decoder2")
    dec = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding="same")(dec) # Max-pooling; (N, T_y, E*K/2)


    dec = tf.keras.layers.Conv1D(filters=hp.decoder_conv_channels, kernel_size=hp.decoder_kernel_size, padding = "same", use_bias = False, name = "decoder2_cnn1")(dec) # Conv1D projections: (N, T_x, E/2)
    dec = bn(dec)
    dec = tf.keras.layers.LeakyReLU(0.2)(dec)

    dec = tf.keras.layers.Conv1D(filters=hp.n_mels, kernel_size=hp.decoder_kernel_size, padding = "same", use_bias = False, name = "decoder2_cnn2")(dec)
    dec = bn(dec)

    dec = tf.keras.layers.Dense(hp.embed_size//2)(dec) # Extra affine transformation for dimensionality sync; (N, T_y, E/2)

    for i in range(4):
        dec = highwaynet(dec, num_units = hp.embed_size//2, naming = "decoder_{}".format(i))
        
    dec = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(hp.embed_size // 2, 
                                                            return_sequences = True,
                                                            name = "decoder2_bidirectional_GRU"))(dec) # output will be automatically concatenated
    
    ##### Create linear outputs
    mag_hat = tf.keras.layers.Dense(1 + hp.n_fft//2)(dec)

    return tf.keras.Model(mel_hat_input, mag_hat)