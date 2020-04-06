# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function, division

import copy
import io
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy import signal
import tensorflow as tf
import unicodedata

from hyperparams import Hyperparams as hp

def get_spectrograms(fpath):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''

    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)


    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''
    Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")


def learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme from tensor2tensor'''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0 # for reduction
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel.reshape((-1, hp.n_mels*hp.r)), mag


def pad_spectrograms(spectrograms):

    t = spectrograms.shape[0]
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0 # for reduction
    padded = tf.pad(spectrograms, [[0, num_paddings], [0, 0]], mode="constant")

    return padded

def pad_melspectrograms(melspectrograms):

    t = melspectrograms.shape[0]
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0 # for reduction
    padded = tf.pad(melspectrograms, [[0, num_paddings], [0, 0]], mode="constant")
    padded = tf.reshape(padded, [-1, hp.n_mels*hp.r])

    return padded


def convert_indexed_slices_to_tensor(idx_slices):
    return tf.scatter_nd(tf.expand_dims(idx_slices.indices, 1),
                         idx_slices.values, idx_slices.dense_shape)


# function for plotting the attention weights
def plot_attention(attention, gs = None):
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(attention, cmap='viridis') 
    
    ##### Record to buf
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    
    if gs is not None:
        if not os.path.exists(r"alignments_1"):
            os.mkdir(r"alignments_1")
        plt.savefig(r"./alignments_1/{}.png".format(gs), format = "png")
    
    buf.seek(0)
    attention_image = tf.image.decode_png(buf.getvalue(), channels = 4)
    attention_image = tf.expand_dims(attention_image, 0)

    plt.close()

    return attention_image


# Function for plotting graph
def plot_graph(value_list):
    
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot(value_list)
    plt.title("test")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    graph_image = tf.image.decode_png(buf.getvalue(), channels = 4)
    graph_image = tf.expand_dims(graph_image, 0)

    plt.close()
    
    return graph_image


# Function for normalizing texts for synthesis
def text_normalize(text, hp):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                        if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text