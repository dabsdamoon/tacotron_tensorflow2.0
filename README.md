# Tacotron with Tensorflow 2.0
Implementation of tacotron (TTS) with Tensorflow 2.0.0 heavily inspired by: </br>

General structure of algorithm: https://github.com/Kyubyong/tacotron </br>
Training in Korean: https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS </br>
Attention: https://www.tensorflow.org/tutorials/text/nmt_with_attention </br>

Notice that I haven't used tensorflow_addon library since it doesn't seem to be fully compatible with Tensorflow >= 2.0.0. </br>
Also, I added an option to choosen between regular and monotonic attention since monotonic attention shows faster convergences in both language cases. For more information about monotonic attention, visit https://arxiv.org/abs/1704.00784 or https://vimeo.com/240608543 if you prefer presentation </br>

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

First, set your parameters (including directory, language, etc) in hyperparams.py. For generating examples, I set "use_monotonic" and "normalize_attention" parameter as True. </br>
Then, you can just run training.py file as follows: </br>
<pre>
<code> 
python training.py 
</code>
</pre>

## Result
To show some example results, I trained with both English and Korean dataset applying Bahdanau monotonic attention with normalization.
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

First, set your parameters in hyperparams.py. Note that you need to set "use_monotonic" and "normalize_attention" parameter as True if you have trained the algorithm in such way. Then, use the function "synthesizing" to generate the sentence you want. </br>

<pre>
<code>
synthesizing("The boy was there when the sun rose", hp)
synthesizing("오늘 점심은 쌀국수 한그릇 먹고싶네요", hp)
</code>
</pre>

Finally, run synthesizing.py with console command:

<pre>
<code> 
python synthesizing.py 
</code>
</pre>

For audio samples, I uploaded synthesized English sentence of "The boy was there when the sun rose" and Korean sentence of "오늘 점심은 쌀국수 한그릇 먹고싶네요" in a folder "sample_synthesis". The algorithm has been trained 77000 steps for English (roughly 40 hours), and 67000 steps for Korean (roughly 15 hours). </br> 

## Notes
* Although I tried to convert Kyubyoung's Tensorflow 1.12 code to Tensorflow 2.0 code as it is, there may be some differences between mine and Kyubyoung's. I'd appreciate if you notice differences and inform me. Also, since I directly implemented Kyubyoung's code, differences from the original paper are also implemented.
* As I have mentioned earlier, training Korean dataset takes quite less time than training English dataset. Thus, if you can understand both languages, you may notice that Korean synthesizing result sounds better than English one. The English result will be better if you spend more time on training.
* Any comments on improving codes or questions are welcome, but it may take some time for me to respond.

April 2020, Dabin Moon
