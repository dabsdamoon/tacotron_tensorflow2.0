# Tacotron Implementation with Tensorflow >= 2.0.0
Implementation of tacotron (TTS) with Tensorflow 2.0.0 heavily inspired by: </br>

General structure of algorithm: https://github.com/Kyubyong/tacotron </br>
Training in Korean: https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS </br>
Attention: https://www.tensorflow.org/tutorials/text/nmt_with_attention </br>

Notice that I haven't used tensorflow_addon library since it doesn't seem to be fully compatible with Tensorflow >= 2.0.0. </br>

## Requirements
* Python=3.7
* tensorflow-gpu >= 2.0.0
* librosa
* tqdm
* matplotlib
* jamo
* unidecode
* inflect

## Data
For English, I've used LJSpeech 1.1 dataset (https://keithito.com/LJ-Speech-Dataset/). </br>
For Korean, I've used KSS dataset (https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset). </br>

## Training

First, set your parameters (including directory, language, etc) in hyperparams.py. </br>
Then train a model using command: </br>
<pre>
<code> 
python training.py 
</code>
</pre>

Results of training English data (LJSpeech) are given below: </br>
![Alt Text](https://github.com/dabsdamoon/gif_save/blob/master/tacotron_English.gif) </br>
![Alt Text](https://github.com/dabsdamoon/gif_save/blob/master/tacotron_English_mel.png)
![Alt Text](https://github.com/dabsdamoon/gif_save/blob/master/tacotron_English_linear.png) </br>
</br>

Results of training Korean data (KSS) are given below: </br>
![Alt Text](https://github.com/dabsdamoon/gif_save/blob/master/tacotron_Korean.gif) </br>
![Alt Text](https://github.com/dabsdamoon/gif_save/blob/master/tacotron_Korean_mel.png)
![Alt Text](https://github.com/dabsdamoon/gif_save/blob/master/tacotron_Korean_linear.png) </br>

Both algorithms have been trained for roughly 15 hours.

## Sample Synthesis

## Result

## Notes
