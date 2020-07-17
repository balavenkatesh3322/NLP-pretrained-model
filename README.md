# NLP-pretrained-model

![NLP logo](https://github.com/balavenkatesh3322/NLP-pretrained-model/blob/master/logo.jpg)

## What is pre-trained Model?
A pre-trained model is a model created by some one else to solve a similar problem. Instead of building a model from scratch to solve a similar problem, we can use the model trained on other problem as a starting point. A pre-trained model may not be 100% accurate in your application.


### Framework

* [Tensorflow](#tensorflow)
* [Keras](#keras)
* [PyTorch](#pytorch)
* [MXNet](#mxnet)
* [Caffe](#caffe)


### Model visualization
You can see visualizations of each model's network architecture by using [Netron](https://github.com/lutzroeder/Netron).

![NLP logo](https://github.com/balavenkatesh3322/NLP-pretrained-model/blob/master/netron.png)

### Tensorflow <a name="tensorflow"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [Chatbot]( https://github.com/Conchylicultor/DeepQA)  | This work tries to reproduce the results of A Neural Conversational Model (aka the Google chatbot). It uses a RNN (seq2seq model) for sentence prediction     | `Tensorflow`
| [Show, Attend and Tell]( https://github.com/yunjey/show-attend-and-tell)  | Attention Based Image Caption Generator.     | `Tensorflow`
| [Seq2seq-Chatbot]( https://github.com/tensorlayer/seq2seq-chatbot)  | Chatbot in 200 lines of code.     | `Tensorflow`
| [Neural Caption Generator]( https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow)  | Implementation of "Show and Tell".     | `Tensorflow`
| [TensorFlow White Paper Notes]( https://github.com/samjabrahams/tensorflow-white-paper-notes)  | Annotated notes and summaries of the TensorFlow white paper, along with SVG figures and links to documentation.     | `Tensorflow`
| [Neural machine translation between the writings of Shakespeare and modern English using TensorFlow]( https://github.com/tokestermw/tensorflow-shakespeare)  | This performs a monolingual translation, going from modern English to Shakespeare and vice-versa.     | `Tensorflow`
| [Mnemonic Descent Method](https://github.com/trigeorgis/mdm)  | Tensorflow implementation of "Mnemonic Descent Method: A recurrent process applied for end-to-end face alignment"     | `Tensorflow`
| [Improved CycleGAN]( https://github.com/luoxier/CycleGAN_Tensorlayer)  | Unpaired Image to Image Translation.     | `Tensorflow`
| [im2im]( https://github.com/zsdonghao/Unsup-Im2Im)  | Unsupervised Image to Image Translation with Generative Adversarial Networks.     | `Tensorflow`
| [DeepSpeech]( https://github.com/tensorflow/models/tree/master/research/deep_speech)  | Automatic speech recognition.     | `Tensorflow`
| [Im2txt]( https://github.com/tensorflow/models/tree/master/research/im2txt)  | Image-to-text neural network for image captioning.     | `Tensorflow`

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### Keras <a name="keras"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [Monolingual and Multilingual Image Captioning]( https://github.com/elliottd/GroundedTranslation)  | This is the source code that accompanies Multilingual Image Description with Neural Sequence Models.     | `Keras`
| [pix2pix]( https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)  | Keras implementation of Image-to-Image Translation with Conditional Adversarial Networks.     | `Keras`
| [DualGAN]( https://github.com/eriklindernoren/Keras-GAN/blob/master/dualgan/dualgan.py)  | Implementation of _DualGAN: Unsupervised Dual Learning for Image-to-Image Translation_.     | `Keras`
| [CycleGAN]( https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py)  | Implementation of _Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_.     | `Keras`

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### PyTorch <a name="pytorch"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [pytorch-CycleGAN-and-pix2pix]( https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  | PyTorch implementation for both unpaired and paired image-to-image translation.     | `PyTorch`
| [vid2vid]( https://github.com/NVIDIA/vid2vid)  | Pytorch implementation of our method for high-resolution (e.g. 2048x1024) photorealistic video-to-video translation.     | `PyTorch`
| [Neural Machine Translation (NMT) System]( https://github.com/OpenNMT/OpenNMT-py)  | This is a Pytorch port of OpenNMT, an open-source (MIT) neural machine translation system. It is designed to be research friendly to try out new ideas in translation, summary, image-to-text, morphology, and many other domains.     | `PyTorch`
| [UNIT]( https://github.com/mingyuliutw/UNIT)  | PyTorch Implementation of our Coupled VAE-GAN algorithm for Unsupervised Image-to-Image Translation.     | `PyTorch`
| [espnet]( https://github.com/espnet/espnet)  | End-to-End Speech Processing Toolkit.  | `PyTorch`
| [TTS]( https://github.com/mozilla/TTS)  | Deep learning for Text2Speech.     | `PyTorch`
| [Neural Sequence labeling model]( https://github.com/jiesutd/NCRFpp)  | Sequence labeling models are quite popular in many NLP tasks, such as Named Entity Recognition (NER), part-of-speech (POS) tagging and word segmentation.    | `PyTorch`
| [UnsupervisedMT]( https://github.com/facebookresearch/UnsupervisedMT)  | Phrase-Based & Neural Unsupervised Machine Translation.     | `PyTorch`
| [waveglow]( https://github.com/NVIDIA/waveglow)  | A Flow-based Generative Network for Speech Synthesis.     | `PyTorch`
| [deepvoice3_pytorch]( https://github.com/r9y9/deepvoice3_pytorch)  | PyTorch implementation of convolutional networks-based text-to-speech synthesis models.     | `PyTorch`
| [deepspeech2]( https://github.com/SeanNaren/deepspeech.pytorch)  | Implementation of DeepSpeech2 using Baidu Warp-CTC. Creates a network based on the DeepSpeech2 architecture, trained with the CTC activation function.    | `PyTorch`
| [pytorch-seq2seq]( https://github.com/IBM/pytorch-seq2seq)  | A framework for sequence-to-sequence (seq2seq) models implemented in PyTorch.     | `PyTorch`
| [loop]( https://github.com/facebookarchive/loop)  | A method to generate speech across multiple speakers.     | `PyTorch`
| [neuraltalk2-pytorch]( https://github.com/ruotianluo/ImageCaptioning.pytorch)  | Image captioning model in pytorch (finetunable cnn in branch with_finetune)     | `PyTorch`
| [seq2seq]( https://github.com/MaximumEntropy/Seq2Seq-PyTorch)  | This repository contains implementations of Sequence to Sequence (Seq2Seq) models in PyTorch.     | `PyTorch`
| [seq2seq.pytorch]( https://github.com/eladhoffer/seq2seq.pytorch)  | Sequence-to-Sequence learning using PyTorch.     | `PyTorch`
| [self-critical.pytorch]( https://github.com/ruotianluo/self-critical.pytorch)  | Self-critical Sequence Training for Image Captioning.     | `PyTorch`
| [Hierarchical Attention Networks for Document Classification]( https://github.com/EdGENetworks/attention-networks-for-classification)  | We know that documents have a hierarchical structure, words combine to form sentences and sentences combine to form documents.     | `PyTorch`
| [nmtpytorch]( https://github.com/lium-lst/nmtpytorch)  | Neural Machine Translation Framework in PyTorch.     | `PyTorch`
| [pix2pix-pytorch]( https://github.com/mrzhu-cool/pix2pix-pytorch)  | PyTorch implementation of "Image-to-Image Translation Using Conditional Adversarial Networks".     | `PyTorch`
| [torch_waveglow]( https://github.com/npuichigo/waveglow)  | A PyTorch implementation of the WaveGlow: A Flow-based Generative Network for Speech Synthesis.     | `PyTorch`
| [Open Source Chatbot with PyTorch]( https://github.com/jinfagang/pytorch_chatbot)  | Aim to build a Marvelous ChatBot.     | `PyTorch`
| [nonauto-nmt]( https://github.com/salesforce/nonauto-nmt)  | PyTorch Implementation of "Non-Autoregressive Neural Machine Translation".     | `PyTorch`
| [tacotron_pytorch]( https://github.com/r9y9/tacotron_pytorch)  | PyTorch implementation of Tacotron speech synthesis model.     | `PyTorch`
| [pytorch-seq2seq-intent-parsing]( https://github.com/spro/pytorch-seq2seq-intent-parsing)  | Intent parsing and slot filling in PyTorch with seq2seq + attention.    | `PyTorch`
| [captionGen]( https://github.com/eladhoffer/captionGen)  | Generate captions for an image using PyTorch.     | `PyTorch`
| [bandit-nmt]( https://github.com/khanhptnk/bandit-nmt)  | This is code repo for our EMNLP 2017 paper "Reinforcement Learning for Bandit Neural Machine Translation with Simulated Human Feedback".     | `PyTorch`
| [Pytorch Poetry Generation]( https://github.com/jhave/pytorch-poetry-generation)  | is a repurposing of http://pytorch.org/: an early release beta software (developed by a consortium led by Facebook and NVIDIA), a deep learning software that puts Python first.     | `PyTorch`
| [translagent]( https://github.com/facebookresearch/translagent)  | Code for Emergent Translation in Multi-Agent Communication.     | `PyTorch`

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***


### MXNet <a name="mxnet"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [MXNMT]( https://github.com/magic282/MXNMT)  | This is an implementation of seq2seq with attention for neural machine translation with MXNet.     | `MXNet`
| [deepspeech]( https://github.com/samsungsds-rnd/deepspeech.mxnet)  | This example based on DeepSpeech2 of Baidu helps you to build Speech-To-Text (STT) models at scale using.     | `MXNet`
| [mxnet-seq2seq]( https://github.com/yoosan/mxnet-seq2seq)  | This project implements the sequence to sequence learning with mxnet for open-domain chatbot.     | `MXNet`

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### Caffe <a name="caffe"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [Speech Recognition](https://github.com/pannous/caffe-speech-recognition)  | Speech Recognition with the caffe deep learning framework.     | `Caffe`

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

## Contributions
Your contributions are always welcome!!
Please have a look at contributing.md

## License

[MIT License](LICENSE)
