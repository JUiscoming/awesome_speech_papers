# About This Repository
이 페이지는 음성 관련 task (음성 인식, 음성 합성  등)를 공부 및 연구하고 싶은 newbie들을 위해 만들어짐. <br>
최대한 페이퍼를 많이 포함하기 보다는 중요하고(citation이 충분히 높고, 신뢰할 만한 기관에서 수행했으며, top 컨퍼런스/에 publish된 페이퍼 위주) 최신자 페이퍼들만 포함하려고 함.(주관적일 수 있음) <br>


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
- 5.Self-Supervised(or Semi-Supervised) Learning for Speech 


<br>

***

# 1. End-to-End Speech Recognition 

- CTC based ASR model
  - If you're new to CTC-based ASR model, you'd better see this blog before reading papers : [post for Distill blog](https://distill.pub/2017/ctc/)

<p align="center"><img src="https://user-images.githubusercontent.com/48202736/99594617-c4020780-2a36-11eb-99af-8f9a200584e1.png"  width="500"></p>
<p align="center">Deep Speech 2 : End-to-End Speech Recognition in English and Mandarin</p> <br>
  
|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2006|ICML|Toronto University|**Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks**|CTC|[paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)||
|2014|||Deep speech: Scaling up end-to-end speech recognition||||
|2016|ICML||**Deep Speech 2 : End-to-End Speech Recognition in English and Mandarin**|CTC-based CNN model|[paper](https://arxiv.org/pdf/1512.02595)|[code(pytorch)](https://github.com/SeanNaren/deepspeech.pytorch)|
|2019|Interspeech|Nvidia|Jasper: An End-to-End Convolutional Neural Acoustic Model||||


- Attention based ASR model
  - If you're new to seq2seq with attention network, you'd better check following things
    - [post for Seq2Seq with Attention Network 1](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
    - [post for Seq2Seq with Attention Network 2](https://distill.pub/2016/augmented-rnns/)
    - [post for Seq2Seq with Attention Network 3](https://guillaumegenthial.github.io/sequence-to-sequence.html)
    - [post for Transformer](http://jalammar.github.io/illustrated-transformer/)

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
|2020||Google|Conformer: Convolution-augmented Transformer for Speech Recognition||[paper](https://arxiv.org/pdf/2005.08100)||
|2020||Google|ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context||[paper](https://arxiv.org/pdf/2005.03191)||

- Hybrid Model

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2012|||aaa||||

<br>

- RNN-T based ASR model

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2012|||Sequence Transduction with Recurrent Neural Networks||||
|2018|ICASSP|Google|**Streaming E2E Speech Recognition For Mobile Devices**||[paper](https://arxiv.org/pdf/1811.06621)||
|2018||Google|Exploring Architectures, Data and Units For Streaming End-to-End Speech Recognition with RNN-Transducer||||
|2019||Google|Improving RNN Transducer Modeling for End-to-End Speech Recognition||||


<br>

- Streaming ASR

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2019||Google|**Two-Pass End-to-End Speech Recognition**||||

<br>
 
***
 
# 2. End-to-End Speech Synthesis 

<br>

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2016||Deepmind|**WaveNet: A Generative Model for Raw Audio**||[paper](https://arxiv.org/pdf/1609.03499)||
|2017|ICLR|-|SampleRNN: An Unconditional End-to-End Neural Audio Generation Model||[paper](https://arxiv.org/pdf/1612.07837)|[code(official)](https://github.com/soroushmehr/sampleRNN_ICLR2017)|
|2017|ICLR|Montreal Univ, CIFAR|Char2Wav: End-to-End Speech Synthesis||[paper](https://openreview.net/pdf?id=B1VWyySKx)||
|2017|PMLR|Baidu Research|Deep Voice: Real-time Neural Text-to-Speech||[paper](http://proceedings.mlr.press/v70/arik17a/arik17a.pdf)||
|2017|NIPS|Baidu Research|Deep Voice 2: Multi-Speaker Neural Text-to-Speech||[paper](https://arxiv.org/pdf/1705.08947)||
|2017||Baidu Research|**Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning**||[paper](https://arxiv.org/pdf/1710.07654)|[code](https://github.com/r9y9/deepvoice3_pytorch)|
|2017||Google|**Tacotron: Towards End-to-End Speech Synthesis**||[paper](https://arxiv.org/pdf/1705.08947)|[code(tensorflow)](https://github.com/Kyubyong/tacotron), [code(pytorch)](https://github.com/r9y9/tacotron_pytorch)|
|2018|||Adversarial Audio Synthesis|GAN|[paper](https://arxiv.org/pdf/1802.04208)|[code(official, tensorflow)](https://github.com/chrisdonahue/wavegan)|
|2019|ICASSP|Nvidia|WaveGlow: a Flow-based Generative Network for Speech Synthesis||[paper](https://arxiv.org/pdf/1811.00002)|[code(official, pytorch)](https://github.com/NVIDIA/waveglow)|
|2019|||Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram||[paper](https://arxiv.org/pdf/1910.11480)||


***

# 3. End-to-End Non-Autoregressive Sequence Generation Model

<pre>
<code>
Non-Autoregressive 모델은 논문이 별로 없기 때문에 기계번역(NMT)/음성인식(STT)/음성합성(STT) 모두 포함하려고 함.
</code>
</pre>

- Non-Autoregressive(NA) ASR, TTS

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2020|Interspeech|-|Mask CTC: Non-Autoregressive End-to-End ASR with CTC and Mask Predict|CTC-based|||
|2020|Interspeech|-|Spike-Triggered Non-Autoregressive Transformer for End-to-End Speech Recognition|CTC-based|||
|2020||-|A Study of Non-autoregressive Model for Sequence Generation||||

- Non-Autoregressive(NA) NMT

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2018|ICLR|The University of Hong Kong|NON-AUTOREGRESSIVE NEURAL MACHINE TRANSLATION||||
|2020||Google|Non-Autoregressive Machine Translation with Latent Alignments||||
|2020||CMU|FlowSeq: Non-Autoregressive Conditional Sequence Generation with Generative Flow||||
|2020||CMU,Berkeley,Peking University|Fast Structured Decoding for Sequence Models||||
|2019|ACL|-|Non-autoregressive Transformer by Position Learning||||
|2020||-|ENGINE: Energy-Based Inference Networks for Non-Autoregressive Machine Translation||||
|2019||University of Tokyo,FAIR,MILA,NYU|Latent-Variable Non-Autoregressive Neural Machine Translation with Deterministic Inference Using a Delta Posterior||||


- Non-Autoregressive(NA) Speech Synthesis, TTS

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2020||Baidu Research|Non-Autoregressive Neural Text-to-Speech||||

<br>

***

# 4. End-to-End Spoken Language Understanding 
<pre>
<code>
기존의 Spoken Language Understanding (SLU) 는 음성을 입력받아 ASR module이 텍스트를 출력하고, 
이를 입력으로 받은 Natural Language Understanding (NLU) module이 감정(emotion)/의도(intent,slot) 등을 결과로 출력했다.

End-to-End Spoken Language Understanding (SLU)란 음성을 입력으로 받아 direct로 결과를 출력함으로써
음성인식 네트워크가 가지고 있는 에러율에 구애받지 않고 fully differentiable 하게 학습하는 것이 목적이다.
</code>
</pre>

* Dataset ( including all speech slu dataset IC/SF/SQA ... )
  - Intent Classification (IC)
  - Spoken Question Answering (SQA)
  - Emotion Recognition (ER)

|task|dataset name|year|conference|title|paper link|dataset link|
|--|---|--|--|------|----|----|
|-|SLURP|2020|EMNLP|SLURP: A Spoken Language Understanding Resource Package|[paper](https://www.aclweb.org/anthology/2020.emnlp-main.588.pdf)|[dataset](https://github.com/pswietojanski/slurp)|
|IC|Fluent Speech Command(FSC)|2019|Interspeech|Speech Model Pre-training for End-to-End Spoken Language Understanding|[paper](https://arxiv.org/pdf/1904.03670)|[dataset](https://github.com/lorenlugosch/end-to-end-SLU)|
|IC|SNIPS|2018||Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces|[paper](https://arxiv.org/pdf/1805.10190)||
|IC|ATIS|1999||The atis spoken language sys- tems pilot corpus|[paper](https://www.aclweb.org/anthology/H90-1021.pdf)||
|IC|TOP or Facebook Semantic Parsing System (FSPS)|2019||Semantic Parsing for Task Oriented Dialog using Hierarchical Representations|[paper](https://arxiv.org/pdf/1810.07942)||
|SQA|Spoken SQuAD(SSQD)|2018|Interspeech|Spoken SQuAD: A Study of Mitigating the Impact of Speech Recognition Errors on Listening Comprehension|[paper](https://arxiv.org/abs/1804.00320)|[dataset](https://github.com/chiahsuan156/Spoken-SQuAD)|
|SQA|Spoken CoQA|2020|-|Towards Data Distillation for End-to-end Spoken Conversational Question Answering|[paper](https://arxiv.org/pdf/2010.08923)|[dataset](https://stanfordnlp.github.io/coqa/)|
|ER|IEMOCAP|2017|-|IEMOCAP: Interactive emotional dyadic motion capture database|[paper](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Busso_2008_5.pdf)|[dataset](https://sail.usc.edu/iemocap/)|
|ER|CMU-MOSEI|2018|-|Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph|[paper](https://www.aclweb.org/anthology/P18-1208.pdf)|[dataset](https://github.com/A2Zadeh/CMU-MultimodalSDK)|

* Intent Classification (IC)

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2019|Interspeech|MILA,CIFAR|Speech Model Pre-training for End-to-End Spoken Language Understanding||[paper](https://arxiv.org/pdf/1904.03670)|[code(official)](https://github.com/lorenlugosch/end-to-end-SLU)|

* Spoken Question Answering (SQA)

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2018|Interspeech||Spoken SQuAD: A Study of Mitigating the Impact of Speech Recognition Errors on Listening Comprehension|dataset|[paper](https://arxiv.org/abs/1804.00320)|[github](https://github.com/chiahsuan156/Spoken-SQuAD)|

* Emotion Recognition (ER)

<br>

***

# 5. Self-Supervised(or Semi-Supervised) Learning for Speech 
<pre>
<code>
Self-Supervised(or Semi-Supervised) Learning 이란 Yann Lecun이 강조했을 만큼 현재 2020년 현재 딥러닝에서 가장 핫 한 주제중 하나이며, 
Label되지 않은 방대한 data를 self-supervised (or semi-supervised) 방법으로 학습하여 입력으로부터 더 좋은 Representation을 찾는 방법이다. 
이렇게 사전 학습(pre-training)된 네트워크는 음성 인식 등 다른 task를 위해 task-specific 하게 미세 조정 (fine-tuning)하여 사용한다.

사전 학습 방법은 AutoEncoder 부터 BERT 까지 다양한 방법으로 기존에 존재했으나 음성에 맞는 방식으로 연구된 논문들이 최근에 제시되어 왔으며, 
이렇게 학습된 네트워크는 scratch 부터 학습한 네트워크보다 더욱 높은 성능을 자랑한다 .
</code>
</pre>

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
