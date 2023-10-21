# CFSD_few-shot

## Description
This repository contains the dataset generation (Matlab) and model file of CFSD+CCKG (Python, Pytorch) for our paper "Few-Shot Learning for Signal Detection in Wideband Spectrograms".

## Abstract
Thanks to great developments in deep learning methods for object detection, an increasing number of researchers have recently introduced related methods to signal detection in spectrograms, and they have obtained remarkable performance. Most existing detection models rely on the availability of abundant labeled training data, but for signal classes with little labeled training data, detection performance can deteriorate significantly. In this paper, a few-shot signal detection model is proposed to solve this problem. The proposed model is pretrained on abundantly labeled base signal classes and aims to detect novel classes given only a few labeled samples. The model is built on a base detector that is designed specifically for signal detection, and a class-specific convolution kernel generator (CCKG) is proposed to generate convolution kernels for predictions of signal center frequency and shape attributes. Benefiting from a three-stage meta-learning procedure, the CCKG can play a significant role with only a few input samples. Comprehensive experiments with a dataset constructed from simulated signals and real backgrounds and a real-world dataset demonstrate that our method yields significantly better performance than direct fine-tuning and popular few-shot object detection methods.

## Set-up

### Dataset generation
You can directly run the 'data_generate/main.m' to generate wideband waves and annotation .txts to folder 'data/'. Some tips:
* Each generated wave contains multiple narrowband signals, and each annotation txt contains the start time, end time (multiplied by 1e7), start frequency, end frequency, and class of each narrowband signal.
* Signal classes contain: '2FSK', '4FSK', '8FSK', 'GMSK', 'PSK', 'Morse', 'AM-DSB', 'AM-USB', 'AM-LSB', 'FM', 'LFM', '8-Tone', '16-Tone', more than in the paper.
* The class 'AM-DSB', 'AM-USB', 'AM-LSB', and 'FM' modulate the songs in the folder 'data_generate/songs', where we put a song as an example.
* Generating wideband waves needs to add real backgrounds, and we put 5 background waves in the folder 'data_generate/backgrounds' as examples. You can add your own background files. We limit the background file to be larger than 5 s and 125 kHz.
### model file
The file 'CFSD_few-shot-master/model.py' contains the model code of CFSD + CCKG in the paper. You can train it according to the three-stage procedure in the paper. For training:
* stage 1 (Base Detector Training) updates all parameters except 'self.meta.parameters()' in the file.
* stage 2 (CCKG Meta-Training) updates 'self.meta_params' in the file.
* stage 3 (Few-shot Fine-Tuning) updates all parameters.

The documents are arranged in a hurry. If you have questions, please contact email liweihao21@nudt.edu.cn. We will upload the complete code as soon as possible after sorting.
