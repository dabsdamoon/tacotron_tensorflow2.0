##### Import modules
import glob
import numpy as np
import subprocess
import tensorflow as tf
import time
from tqdm import tqdm

##### Import custom module
from hyperparams import Hyperparams as hp
from utils import *
from dataloader import *
from modules import *
from kor_text.symbols import symbols

##### Define models
def training(dataloader, hp):
    
    ##### Define encoder
    encoder = get_encoder(hp)

    ##### Define decoder1 (attention decoder)

    # Get attention mechanism based on monotonic and normalize conditions
    if hp.use_monotonic and hp.normalize_attention:
        attention_mechanism = BahdanauMonotonicAttention(hp.embed_size, normalize = True)

    elif hp.use_monotonic and not hp.normalize_attention:
        attention_mechanism = BahdanauMonotonicAttention(hp.embed_size)

    elif not hp.use_monotonic and hp.normalize_attention:
        attention_mechanism = BahdanauAttention(hp.embed_size, normalize = True)

    elif not hp.use_monotonic and not hp.normalize_attention:
        attention_mechanism = BahdanauAttention(hp.embed_size)

    decoder1 = AttentionDecoder(attention_mechanism, 
                                hp)

    ##### Define decoder2
    decoder2 = get_decoder2(hp)

    ##### Restore the model; else, create new one.
    if hp.restore:
        indexfile = glob.glob(os.path.join(hp.model_dir, "encoder/*.index"))
        indexlist = [int(i_f.split(r"_")[-1].split(r".")[0]) for i_f in indexfile]
        step_index = max(indexlist)
        
        n_epoch = 1 + (step_index // dataloader.total_batch_num)
        
        encoder.load_weights(os.path.join(hp.model_dir, "encoder/weights_{}".format(step_index)))
        decoder1.load_weights(os.path.join(hp.model_dir, "decoder1/weights_{}".format(step_index)))
        decoder2.load_weights(os.path.join(hp.model_dir, "decoder2/weights_{}".format(step_index)))

        print("===== Model restoration complete (step_index = {} // n_epoch = {})".format(step_index, n_epoch))

    else:
        subprocess.run(["rm", "-rf", "./{}".format(hp.log_dir)])
        subprocess.run(["rm", "-rf", "./{}".format(hp.model_dir)])

        n_epoch = 1
        step_index = 0
        
    train_summary_writer = tf.summary.create_file_writer(hp.log_dir)
    
    ##### Training
    while 1:
    
        print("===== Epoch: {}".format(n_epoch))

        ##### Record loss by batch
        mel_loss = [] 
        mag_loss = []
        start = time.time()

        batch_index = 0

        for x, y, z in dataloader.loader:

            batch_index += 1

            lr = learning_rate_decay(hp.lr, global_step=step_index)
            opt = tf.keras.optimizers.Adam(lr, clipnorm=5.0)
            attention_plot = np.zeros((len(x), 
                                       x.shape[1], # Timestep of text  
                                       y.shape[1])) # Timestep of mel

            with tf.GradientTape() as tape:

                ##### Compute memory and initla state for decoder
                memory, memory_state, memory_mask = encoder(x)

                plot_length = tf.reduce_sum(tf.cast(memory_mask, tf.int32), axis = 1).numpy()

                ##### Define decoder input and initial state
                decoder1_input_init = tf.zeros_like(y[:,0,:])
                decoder1_attn_state = decoder1.attention_cell.get_initial_state(memory) 
                decoder1_decoder_state = decoder1.decoder_cell.get_initial_state(memory)
                initial_alignments = decoder1._initial_alignments(x.shape[0], 
                                                                  x.shape[1], 
                                                                  dtype = tf.float32)

                # Create a list of states
                decoder1_state = [decoder1_attn_state, 
                                  decoder1_decoder_state, 
                                  initial_alignments, 
                                  memory, 
                                  memory_mask]

                decoder1_input = decoder1_input_init
                decoder1_output = []            

                for t in range(y.shape[1]): # Iterate based on timestep of the batch (the longest timestep within batch)

                    d, decoder1_state = decoder1(decoder1_input,
                                                        decoder1_state)

                    alignments = decoder1_state[2]

                    ##### Appending mel-spectogram feature prediction
                    decoder1_output.append(d)

                    ##### Upldate input and attention plot
                    decoder1_input = y[:,t,:]
                    attention_plot[:, :, t] = alignments.numpy()

                ##### Calculate losses (mel-spectrogram)
                y_hat = tf.concat(decoder1_output, axis = 1)
                loss1 = tf.reduce_mean(tf.abs(y_hat - y)) # Mel-spectrogram loss

                z_hat = decoder2(y_hat)
                loss2 = tf.reduce_mean(tf.abs(z_hat - z)) # Linear-spectrogram loss

                loss = loss1 + loss2    

            ##### Update gradients
            variables = encoder.trainable_variables + decoder1.trainable_variables + decoder2.trainable_variables
            gradients = tape.gradient(loss, variables)
            opt.apply_gradients(zip(gradients, variables))

            ##### Update step index
            step_index += 1            

            ##### Draw norm
            """
            This is a plot of gradient norm values of every layers.
            With this graph, one can see whether gradients are flowing well throughout the model.
            """
            grad_norm = []

            for i, grads in enumerate(gradients):

                if i == 0:
                    idxslice_to_tensor = convert_indexed_slices_to_tensor(grads)
                    grad_norm.append(np.linalg.norm(idxslice_to_tensor.numpy()))

                else:

                    if len(grads.get_shape()) > 0:
                        grad_norm.append(np.linalg.norm(grads[0]))

                    else: # Case where gradient value is just a single scalar
                        grad_norm.append(np.abs(grads))

            ##### Record summaryfor tensorboard
            if step_index % 100 == 0:

                with train_summary_writer.as_default():

                    # Losses
                    tf.summary.scalar("Mel-Spectrogram Loss", loss1, step = step_index)
                    tf.summary.scalar("Linear Spectrogram Loss", loss2, step = step_index)

                    # Learning rate
                    tf.summary.scalar("Learning Rate", lr, step = step_index)
                    tf.summary.image("Attention (whole)", plot_attention(attention_plot[0]), step = step_index)
                    tf.summary.image("Gradient Norm Plot", plot_graph(grad_norm), step = step_index)

                    tf.summary.histogram("Gradient Norm", tf.convert_to_tensor(grad_norm), step = step_index)

                    tf.summary.audio("Test Voice", spectrogram2wav(z_hat[0].numpy()).reshape((1,-1, 1)), sample_rate=hp.sr, step = step_index)

                    print('Time taken for every 100th batch: {} secn (total: {} batches)'.format(time.time() - start, step_index))
                    start = time.time()

            ##### Save model weights
            if step_index % 1000 == 0:

                model_list = ["encoder", "decoder1", "decoder2"]

                ##### Creating directory
                if not os.path.exists(hp.model_dir):
                    os.mkdir(hp.model_dir)

                for m in model_list:
                    if not os.path.exists(os.path.join(hp.model_dir, m)):
                        os.mkdir(os.path.join(hp.model_dir, m))

                encoder.save_weights(os.path.join(hp.model_dir, "encoder/weights_{}".format(step_index)))
                decoder1.save_weights(os.path.join(hp.model_dir, "decoder1/weights_{}".format(step_index)))
                decoder2.save_weights(os.path.join(hp.model_dir, "decoder2/weights_{}".format(step_index)))

        n_epoch += 1

if __name__ == "__main__":

    ##### Choose the source
    # hp.source = "korean"
    
    # Set hp.vocab; originally for LJSpeech (English) dataset. If Korean, symbols need to be changed
    if hp.source == "LJSpeech":
        dl = DataLoader(hp)
    else:
        hp.vocab = symbols
        dl = DataLoader(hp)

    ##### Choose type of attention mechanism; I chose monotonic normalized attention mechanism since it seems faster than regular one.
    hp.use_monotonic = True
    hp.normalize_attention = True
    
    ##### Train
    training(dl, hp)
