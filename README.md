# About This Repository

***
This repository is for those who want to study Speech tasks such as Speech Recognition, Speecn Synthesis, Spoken Language Understanding and so on).
I did not try to survey as many papers as possible but the most crucial papers (especially recently published papers) by my standards.
***

<br>

***
이 페이지는 음성 관련 task (음성 인식, 음성 합성  등)를 공부 및 연구하고 싶은 newbie들을 위해 만들어졌습니다. 
최대한 페이퍼를 많이 포함하기 보다는 중요하고(citation이 충분히 높고, 신뢰할 만한 기관에서 수행했으며, top 컨퍼런스에 publish된 페이퍼 위주) 최신자 페이퍼들만 포함하려고 합니다. (주관적일 수 있음) 
***

### temporal (training schemes or undefined)
- State-of-the-Art Speech Recognition Using Multi-Stream Self-Attention With Dilated 1D Convolutions [paper](https://arxiv.org/pdf/1910.00716)



## Index
- 1.End-to-End Speech Recognition papers
  - CTC-based ASR papers
  - Attention-based ASR papers
  - Hybrid ASR papers
  - RNN-T based ASR papers
  - Streaming ASR papers
  
- 2.End-to-End Speech Synthesis papers

- 3.End-to-End Non-Autoregressive Sequence Generation papers
  - ASR
  - NMT
  - TTS

- 4.End-to-End Spoken Language Understanding 
  - Intent Classification papers
  - Spoken Question Answering papers
  - Speech Emotion Recognition papers

- 5.Front-End Learning for Speech 

- 6.Self-Supervised(or Semi-Supervised) Learning for Speech 

- 7.Training Schemes for Speech recognition


- TBC
  - Voice Conversion
  - Speaker Identification
  - MIR ?
  - Rescoring
  - Speech Translation


<br>

***

<br>

# 1. End-to-End Speech Recognition 

## **1.1 CTC based ASR model**
- If you're new to CTC-based ASR model, you'd better see this blog before reading papers : [post for CTC from Distill blog](https://distill.pub/2017/ctc/)
  - additional : **For Korean : [link1](https://m.blog.naver.com/PostView.nhn?blogId=sogangori&logNo=221183469708&proxyReferer=https:%2F%2Fwww.google.com%2F), [link2](https://ratsgo.github.io/speechbook/docs/neuralam/ctc)**

<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99594617-c4020780-2a36-11eb-99af-8f9a200584e1.png"></p>
<p align="center">< Deep Speech 2 : End-to-End Speech Recognition in English and Mandarin ></p> <br>
  
|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2006|ICML|Toronto University|**Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks**|CTC|[paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)||
|2014|ICML||Towards End-To-End Speech Recognition with Recurrent Neural Network|LSTM-based CTC model|[paper](http://proceedings.mlr.press/v32/graves14.pdf)||
|2014|||Deep speech: Scaling up end-to-end speech recognition||||
|2016|ICML||**Deep Speech 2 : End-to-End Speech Recognition in English and Mandarin**|CNN-based CTC model|[paper](https://arxiv.org/pdf/1512.02595)|[code(pytorch)](https://github.com/SeanNaren/deepspeech.pytorch)|
|2016||FAIR|**Wav2Letter: an End-to-End ConvNet-based Speech Recognition System**|CNN-based CTC model|[paper](https://arxiv.org/pdf/1609.03193)||
|2019|Interspeech|Nvidia|Jasper: An End-to-End Convolutional Neural Acoustic Model||||
|2019||Nvidia|**Quartznet: Deep automatic speech recognition with 1d time-channel separable convolutions**||||

<br>

## **1.2 Attention based ASR model**
- If you're new to seq2seq with attention network, you'd better check following things
  - [post for Seq2Seq with Attention Network 1 from lillog](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
  - [post for Seq2Seq with Attention Network 2 from distill](https://distill.pub/2016/augmented-rnns/)
  - [post for Seq2Seq with Attention Network 3](https://guillaumegenthial.github.io/sequence-to-sequence.html)
  - [post for Transformer from Jay Alammar](http://jalammar.github.io/illustrated-transformer/)

- for koreeans : [link](https://seunghyunseo.github.io/speech_recognition/2021/02/05/LAS/)
    
<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99594615-c3697100-2a36-11eb-9d79-fe819d8958d2.png"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99594613-c2d0da80-2a36-11eb-948f-9790dc9365f6.png"></p>
<p align="center">< Listen, Attend and Spell ></p> <br>

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2008|||Supervised Sequence Labelling with Recurrent Neural Networks||||
|2014|ICML||Towards End-to-End Speech Recognition with Recurrent Neural Networks||||
|2015|NIPS||Attention-Based Models for Speech Recognition|Seq2Seq|||
|2015|ICASSP|Google|**Listen, Attend and Spell**|Seq2Seq|[paper](https://arxiv.org/pdf/1508.01211)|[code(pytorch)](https://github.com/clovaai/ClovaCall)|
|2016|||End-to-End Attention-based Large Vocabulary Speech Recognition||||
|2017|ICLR||**Monotonic Chunkwise Attention**||||
|2018|ICASSP||**Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition**||||
|2019|||Listen, Attend, Spell and Adapt: Speaker Adapted Sequence-to-Sequence ASR||||
|2019|||**A Comparative Study on Transformer vs RNN in Speech Applications**||[paper](https://arxiv.org/pdf/1909.06317)||
|2019|||**End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures**||[paper](https://arxiv.org/pdf/1911.08460)||
|2020||Google|Conformer: Convolution-augmented Transformer for Speech Recognition||[paper](https://arxiv.org/pdf/2005.08100)||

## **1.3 Hybrid Model**

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2019|||Transformer-based Acoustic Modeling for Hybrid Speech Recognition||[paper](https://arxiv.org/pdf/1910.09799)||

<br>

## **1.4 RNN-T based ASR model**

<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99594611-c2384400-2a36-11eb-99bc-c4a9fd426e55.png"></p>
<p align="center">< Streaming E2E Speech Recognition For Mobile Devices ></p> <br>

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2012|||Sequence Transduction with Recurrent Neural Networks||||
|2018|ICASSP|Google|**Streaming E2E Speech Recognition For Mobile Devices**||[paper](https://arxiv.org/pdf/1811.06621)||
|2018||Google|Exploring Architectures, Data and Units For Streaming End-to-End Speech Recognition with RNN-Transducer||||
|2019||Google|Improving RNN Transducer Modeling for End-to-End Speech Recognition||||
|2019||-|Self-Attention Transducers for End-to-End Speech Recognition||||
|2020|ICASSP|-|**Transformer Transducer: A Streamable Speech Recognition Model With Transformer Encoders And RNN-T Loss**||||
|2020|ICASSP|-|A Streaming On-Device End-to-End Model Surpassing Server-Side Conventional Model Quality and Latency||||
|2021|ICASSP|-|FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization||||
|2021|ICASSP|-|Improved Neural Language Model Fusion for Streaming Recurrent Neural Network Transducer||||
|2020||Google|ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context||[paper](https://arxiv.org/pdf/2005.03191)||

<br>

## **1.5 Streaming ASR**

```
사실 1.4의 RNN-T가 곧 Straeming ASR을 위해 디자인 되었는데 그게 그거 아니냐 라고 할 수도 있지만,
RNN-T 이외에도, 어텐션 기반 seq2seq모델만으로 하려는 시도가 있었고, seq2seq 와 RNN-T를 합친 모델 등도 있었기 때문에
따로 빼서 서브섹션을 하나 더 만들었습니다.
```

<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99951915-299d1d80-2dc2-11eb-916c-51ea46602ca6.png"></p>
<p align="center">< Two-Pass End-to-End Speech Recognition ></p> <br>

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2019||Google|**Two-Pass End-to-End Speech Recognition**|LAS+RNN-T|[paper](https://arxiv.org/pdf/1908.10992)||

<br>

## **1.5 ASR Rescoring / Spelling Correction (2-pass decoding)**

```
temporal
```

- This is from [link](https://github.com/SeunghyunSEO/speech-recognition-papers)

|year|conference|research organization|title|model|task|link|code|
|--|--|--|------|---|--|--|--|
|2019|||Automatic Speech Recognition Errors Detection and Correction|||||
|2019|||A Spelling Correction Model For E2E Speech Recognition|||||
|2019|||An Empirical Study Of Efficient ASR Rescoring With Transformers|||||
|2019|||Automatic Spelling Correction with Transformer for CTC-based End-to-End Speech Recognition|||||
|2019|||Correction of Automatic Speech Recognition with Transformer Sequence-To-Sequence Model|||||
|2019|||Effective Sentence Scoring Method Using BERT for Speech Recognition||asr|||
|2019|||Spelling Error Correction with Soft-Masked BERT||nlp|||
|2019|||Parallel Rescoring with Transformer for Streaming On-Device Speech Recognition||asr|||

 
***

<br>
 
# 2. End-to-End Speech Synthesis 

<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99596501-bb5f0080-2a39-11eb-8e70-0411e2b4926d.png"></p>
<p align="center">< Tacotron: Towards End-to-End Speech Synthesis ></p> <br>

<br>

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2016||Deepmind|**WaveNet: A Generative Model for Raw Audio**||[paper](https://arxiv.org/pdf/1609.03499)||
|2017|ICLR|-|SampleRNN: An Unconditional End-to-End Neural Audio Generation Model||[paper](https://arxiv.org/pdf/1612.07837)|[code(official)](https://github.com/soroushmehr/sampleRNN_ICLR2017)|
|2017|ICLR|Montreal Univ, CIFAR|Char2Wav: End-to-End Speech Synthesis||[paper](https://openreview.net/pdf?id=B1VWyySKx)||
|2017|PMLR|Baidu Research|Deep Voice: Real-time Neural Text-to-Speech||[paper](http://proceedings.mlr.press/v70/arik17a/arik17a.pdf)||
|2017|NIPS|Baidu Research|Deep Voice 2: Multi-Speaker Neural Text-to-Speech||[paper](https://arxiv.org/pdf/1705.08947)||
|2017||Baidu Research|**Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning**||[paper](https://arxiv.org/pdf/1710.07654)|[code](https://github.com/r9y9/deepvoice3_pytorch)|
|2017||Google|**Tacotron: Towards End-to-End Speech Synthesis**||[paper](https://arxiv.org/pdf/1703.10135)|[code(tensorflow)](https://github.com/Kyubyong/tacotron), [code(pytorch)](https://github.com/r9y9/tacotron_pytorch)|
|2017|ICML||Emotional End-to-End Neural Speech Synthesizer||||
|2018|ICML||**Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron**||||
|2018|ICML||**Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis**||||
|2021|ICLR|Google Research|Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling||[paper](https://arxiv.org/pdf/2010.04301v1.pdf)||
|2018|||Adversarial Audio Synthesis|GAN|[paper](https://arxiv.org/pdf/1802.04208)|[code(official, tensorflow)](https://github.com/chrisdonahue/wavegan)|
|2019|ICASSP|Nvidia|WaveGlow: a Flow-based Generative Network for Speech Synthesis||[paper](https://arxiv.org/pdf/1811.00002)|[code(official, pytorch)](https://github.com/NVIDIA/waveglow)|
|2019|||Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram||[paper](https://arxiv.org/pdf/1910.11480)||
|2019|NIPS|NVIDIA|FastSpeech: Fast, Robust and Controllable Text to Speech||[paper](https://arxiv.org/pdf/1905.09263)||
|2020|-|NVIDIA|**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**||[paper](https://arxiv.org/pdf/2006.04558)||
|2020|NIPS|Kakao Enterprise, SNU|Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search||[paper](https://arxiv.org/pdf/2005.11129)||
|2020|ICASSP||Flow-TTS: A Non-Autoregressive Network for Text to Speech Based on Flow||[paper](https://ieeexplore.ieee.org/document/9054484)||
|2019|AAAI||Neural Speech Synthesis with Transformer Network||[paper](https://arxiv.org/pdf/1809.08895)||
|2017|||Parallel WaveNet: Fast High-Fidelity Speech Synthesis||||
|2018||-|WaveGlow: A Flow-based Generative Network for Speech Synthesis||||
|2020|ICASSP||Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis||||


***

<br>

# 3. End-to-End Non-Autoregressive Sequence Generation Model

```
Non-Autoregressive 모델은 논문이 별로 없기 때문에 기계번역(NMT)/음성인식(STT)/음성합성(STT) 모두 포함하려고 함.
```
## **3.1 Non-Autoregressive(NA) NMT**

<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99598419-d717d600-2a3c-11eb-812b-f45c8bb815d5.png"></p>
<p align="center">< NON-AUTOREGRESSIVE NEURAL MACHINE TRANSLATION ></p> <br>
<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99598089-5b1d8e00-2a3c-11eb-8196-a629bcda9338.png"></p>
<p align="center">< Latent-Variable Non-Autoregressive Neural Machine Translation with Deterministic Inference Using a Delta Posterior ></p> <br>

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2018|ICLR|The University of Hong Kong|NON-AUTOREGRESSIVE NEURAL MACHINE TRANSLATION||||
|2020||Google|Non-Autoregressive Machine Translation with Latent Alignments||||
|2020||CMU|FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow||||
|2020||CMU,Berkeley,Peking University|Fast Structured Decoding for Sequence Models||||
|2019|ACL|-|Non-autoregressive Transformer by Position Learning||||
|2020||-|ENGINE: Energy-Based Inference Networks for Non-Autoregressive Machine Translation||||
|2019||University of Tokyo,FAIR,MILA,NYU|Latent-Variable Non-Autoregressive Neural Machine Translation with Deterministic Inference Using a Delta Posterior||||

<br>

## **3.2 Non-Autoregressive(NA) ASR (STT)**

<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99598098-5d7fe800-2a3c-11eb-8bc0-d668d8ed34f5.png"></p>
<p align="center">< Mask CTC: Non-Autoregressive End-to-End ASR with CTC and Mask Predict ></p> <br>
<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99598101-5eb11500-2a3c-11eb-9ff9-1ea99aa0c6fd.png"></p>
<p align="center">< Spike-Triggered Non-Autoregressive Transformer for End-to-End Speech Recognition ></p> <br>

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2020|Interspeech|-|Mask CTC: Non-Autoregressive End-to-End ASR with CTC and Mask Predict|CTC-based|||
|2020|Interspeech|-|Spike-Triggered Non-Autoregressive Transformer for End-to-End Speech Recognition|CTC-based|||
|2020||-|A Study of Non-autoregressive Model for Sequence Generation||||

<br>

## **3.3 Non-Autoregressive(NA) Speech Synthesis (TTS)**

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2020||Baidu Research|Non-Autoregressive Neural Text-to-Speech||||

<br>

***

<br>

# 4. End-to-End Spoken Language Understanding 
```
기존의 Spoken Language Understanding (SLU) 는 음성을 입력받아 ASR module이 텍스트를 출력하고, 
이를 입력으로 받은 Natural Language Understanding (NLU) module이 감정(emotion)/의도(intent,slot) 등을 결과로 출력했다.

End-to-End Spoken Language Understanding (SLU)란 음성을 입력으로 받아 direct로 결과를 출력함으로써
음성인식 네트워크가 가지고 있는 에러율에 구애받지 않고 fully differentiable 하게 학습하는 것이 목적이다.
```

<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99597193-f7df2c00-2a3a-11eb-8c93-3e2b9a6698eb.png"></p>
<p align="center">( Conventional Pipeline for Spoken Language Understanding ( ASR -> NLU ) )</p> <br>
<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99597198-f9105900-2a3a-11eb-9e04-c1646ed3188f.png"></p>
<p align="center">( End-to-End Spoken Language Understanding Network )</p> <br>
<p align="center">< Towards End-to-end Spoken Language Understanding ></p> <br>


## **4.1 Dataset ( including all speech slu dataset IC/SF/SQA ... )**
  - Intent Classification (IC)
  - Spoken Question Answering (SQA)
  - Emotion Recognition (ER)

|task|dataset name|language|year|conference|title|paper link|dataset link|
|--|---|--|--|--|------|----|----|
|-|SLURP|english|2020|EMNLP|SLURP: A Spoken Language Understanding Resource Package|[paper](https://www.aclweb.org/anthology/2020.emnlp-main.588.pdf)|[dataset](https://github.com/pswietojanski/slurp)|
|IC|Fluent Speech Command(FSC)|english|2019|Interspeech|Speech Model Pre-training for End-to-End Spoken Language Understanding|[paper](https://arxiv.org/pdf/1904.03670)|[dataset](https://github.com/lorenlugosch/end-to-end-SLU)|
|IC|SNIPS|english|2018||Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces|[paper](https://arxiv.org/pdf/1805.10190)||
|IC|ATIS|english|1999||The atis spoken language sys- tems pilot corpus|[paper](https://www.aclweb.org/anthology/H90-1021.pdf)||
|IC|TOP or Facebook Semantic Parsing System (FSPS)|2019||Semantic Parsing for Task Oriented Dialog using Hierarchical Representations|[paper](https://arxiv.org/pdf/1810.07942)||
|SQA|Spoken SQuAD(SSQD)|english|2018|Interspeech|Spoken SQuAD: A Study of Mitigating the Impact of Speech Recognition Errors on Listening Comprehension|[paper](https://arxiv.org/abs/1804.00320)|[dataset](https://github.com/chiahsuan156/Spoken-SQuAD)|
|SQA|Spoken CoQA|english|2020|-|Towards Data Distillation for End-to-end Spoken Conversational Question Answering|[paper](https://arxiv.org/pdf/2010.08923)|[dataset](https://stanfordnlp.github.io/coqa/)|
|SQA|Odsaqa|chinese|20-|-|Odsqa: Open-domain spoken question answering dataset|-|-|
|ER|IEMOCAP|english|2017|-|IEMOCAP: Interactive emotional dyadic motion capture database|[paper](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Busso_2008_5.pdf)|[dataset](https://sail.usc.edu/iemocap/)|
|ER|CMU-MOSEI|english|2018|-|Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph|[paper](https://www.aclweb.org/anthology/P18-1208.pdf)|[dataset](https://github.com/A2Zadeh/CMU-MultimodalSDK)|


<br>

## **4.2 Intent Classification (IC)**

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2018|ICASSP|Facebook, MILA|Towards End-to-end Spoken Language Understanding||[paper](https://arxiv.org/pdf/1802.08395)||
|2019|Interspeech|MILA,CIFAR|Speech Model Pre-training for End-to-End Spoken Language Understanding||[paper](https://arxiv.org/pdf/1904.03670)|[code(official)](https://github.com/lorenlugosch/end-to-end-SLU)|

<br>

## **4.3 Spoken Question Answering (SQA)**

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2018|Interspeech||Spoken SQuAD: A Study of Mitigating the Impact of Speech Recognition Errors on Listening Comprehension|dataset|[paper](https://arxiv.org/abs/1804.00320)|[github](https://github.com/chiahsuan156/Spoken-SQuAD)|

<br>

## **4.4 Emotion Recognition (ER)**

<br>

***

<br>


# 5. Self-Supervised(or Semi-Supervised) Learning for Speech 


# 6. Self-Supervised(or Semi-Supervised) Learning for Speech 
```
Self-Supervised(or Semi-Supervised) Learning 이란 Yann Lecun이 강조했을 만큼 현재 2020년 현재 딥러닝에서 가장 핫 한 주제중 하나이며, 
Label되지 않은 방대한 data를 self-supervised (or semi-supervised) 방법으로 학습하여 입력으로부터 더 좋은 Representation을 찾는 방법이다. 
이렇게 사전 학습(pre-training)된 네트워크는 음성 인식 등 다른 task를 위해 task-specific 하게 미세 조정 (fine-tuning)하여 사용한다.

사전 학습 방법은 AutoEncoder 부터 BERT 까지 다양한 방법으로 기존에 존재했으나 음성에 맞는 방식으로 연구된 논문들이 최근에 제시되어 왔으며, 
이렇게 학습된 네트워크는 scratch 부터 학습한 네트워크보다 더욱 높은 성능을 자랑한다 .
```

<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99594604-bfd5ea00-2a36-11eb-9c10-4e7385221408.png"></p>
<p align="center">< wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations ></p> <br>

|year|conference|research organization|title|link|code|
|--|--|--|------|--|--|
|2019|-|Facebook AI Research (FAIR)|**wav2vec: Unsupervised Pre-training for Speech Recognition**|[paper](https://arxiv.org/pdf/1904.05862)|[code(official)](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec)|
|2019|-|FAIR|Unsupervised Cross-lingual Representation Learning at Scale|||
|2019|ICLR|FAIR|vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations|[paper](https://arxiv.org/pdf/1910.05453)|[code(official)](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec)|
|2020|-|FAIR|**wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**|[paper](https://arxiv.org/pdf/2006.11477)|[code(official)](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec)|
|2020|-|FAIR|Unsupervised Cross-lingual Representation Learning for Speech Recognition|[paper](https://arxiv.org/pdf/2006.13979)||
|2019|-|Deepmind|Learning robust and multilingual speech representations|[paper](https://arxiv.org/pdf/2001.11128)||
|-|-||SpeechBERT: An Audio-and-text Jointly Learned Language Model for End-to-end Spoken Question Answering|[paper](https://arxiv.org/pdf/1910.11559)||
|-|-||Self-Supervised Representations Improve End-to-End Speech Translation|[paper](https://arxiv.org/pdf/1508.01211)||
|-|-||Unsupervised Pretraining Transfers Well Across Languages|||
|-|-||Learning Problem-agnostic Speech Representations from Multiple Self-supervised Tasks|||
|-|-||Learning robust and multilingual speech representations|||
|-|-||Problem-Agnostic Speech Embeddings for Multi-Speaker Text-to-Speech with SampleRNN|||
|2020|-|MIT CSAIL|SEMI-SUPERVISED SPEECH-LANGUAGE JOINT PRE- TRAINING FOR SPOKEN LANGUAGE UNDERSTANDING|[paper](https://arxiv.org/pdf/2010.02295)||



# 7. Training Schemes for Speech Tasks 

|year|conference|research organization|title|link|code|
|--|--|--|Don't decay the learning rate, Increase the batch size|[paper](https://openreview.net/pdf?id=B1Yy1BxCZ)|--|
|2018|--|--|when does label smoothing help?|[paper](https://arxiv.org/abs/1906.02629)|--|
|--|--|--|Bag of Tricks for Efficient Text Classification|[paper](https://arxiv.org/abs/1607.01759)|--|
|2019|--|Google|SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition|[paper](https://arxiv.org/pdf/1904.08779)|--|
