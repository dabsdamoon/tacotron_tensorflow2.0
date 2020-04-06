# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Originally by kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron

Edited by Dabin Moon (dabsdamoon@neowiz.com)
https://github.com/dabsdamoon

'''
class Hyperparams:

    """
    Hyperparameters
    """
    
    ##### Whether to restore the most recent model
    restore = False
    # restore = True # if True, it will continue to learn from the latest save checkpoint

    # pipeline
    prepro = False  # if True, run `python prepro.py` first before running `python train.py`.
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding E: End of Sentence

    # data
    source = "LJSpeech"
    cleaners = "korean_cleaners" # cleaner to be used for tokenizing Korean

    data = "/home/dabsdamoon/w/projects/tacotron/LJ_data/LJSpeech-1.1"
    # data_korean = "/home/dabsdamoon/w/projects/tacotron2/korean_speaker_npy_data"
    data_korean = "/home/dabsdamoon/w/projects/tacotron2/korean-single-speaker-speech-dataset"

    test_data = "harvard_sentences.txt"
    max_duration = 10.0

    # signal processing
    sr = 22050 # Sample rate.
    n_fft = 2048 # fft points (samples)
    frame_shift = 0.0125 # seconds
    frame_length = 0.05 # seconds
    hop_length = int(sr*frame_shift) # samples.
    win_length = int(sr*frame_length) # samples.
    n_mels = 80 # Number of Mel banks to generate
    power = 1.2 # Exponent for amplifying the predicted magnitude
    n_iter = 50 # Number of inversion iterations
    preemphasis = .97 # or None
    max_db = 100
    ref_db = 20

    # model
    embed_size = 256 # alias = E

    encoder_num_banks = 16
    encoder_conv_channels = 256
    encoder_prenet_size = encoder_conv_channels # should be equal to the encoder convolution channels since it's convolutioning with stride = 2
    encoder_kernel_size = 5

    decoder_num_banks = 8
    decoder_conv_channels = 256
    decoder_kernel_size = 5

    num_highwaynet_blocks = 4
    r = 5 # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5

    # Attention related parameters
    normalize_attention = False
    use_monotonic = False

    # training scheme
    lr = 0.001 # Initial learning rate.
    log_dir = "log/8882"
    model_dir = "model_saved"
    batch_size = 8
    batches_per_group = 32

    # parameters used when preprocessing Korean data
    min_tokens = 30  # originally 50 30 is good for korean; set the mininum length of Korean text to be used for training
    min_n_frame = 30*r  # min_n_frame = reduction_factor * min_iters
    max_n_frame = 200*r
    frame_shift_ms=None      # hop_size=  sample_rate *  frame_shift_ms / 1000
