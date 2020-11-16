# About This Repository
이 페이지는 음성 관련 task (음성 인식,음성 합성 등)를 공부 및 연구하고 싶은 사람들을 위해 만들어졌습니다. 

<br>

# End-to-End Speech Recognition (음성 인식) 
- CTC based ASR model
|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2006|ICML|Toronto University|Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks|CTC|[paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)||
|2014|||Deep speech: Scaling up end-to-end speech recognition||||
|2016|ICML||Deep Speech 2 : End-to-End Speech Recognition in English and Mandarin|CTC-based CNN model|||
|2019|Interspeech|Nvidia|Jasper: An End-to-End Convolutional Neural Acoustic Model|CTC-based CNN model|||

- Attention based ASR model
|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2008|||Supervised Sequence Labelling with Recurrent Neural Networks||||
|2014|ICML||Towards End-to-End Speech Recognition with Recurrent Neural Networks||||
|2015|NIPS||Attention-Based Models for Speech Recognition|Seq2Seq|||
|2015|ICASSP|Google|Listen, Attend and Spell|Seq2Seq|[paper](https://arxiv.org/pdf/1508.01211)|[code](https://github.com/clovaai/ClovaCall)|
|2016|||End-to-End Attention-based Large Vocabulary Speech Recognition||||
|2017|||Monotonic Chunkwise Attention||||
|2018|||Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition||||
|2019|||Listen, Attend, Spell and Adapt: Speaker Adapted Sequence-to-Sequence ASR||||



<br>
 
# End-to-End Speech Synthesis (음성 합성)

<br>

# End-to-End Non-Autoregressive Speech Recognition 

<pre>
<code>
tmp
</code>
</pre>

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2020|Interspeech|-|Mask CTC: Non-Autoregressive End-to-End ASR with CTC and Mask Predict|CTC-based|||
|2020|Interspeech|-|Spike-Triggered Non-Autoregressive Transformer for End-to-End Speech Recognition|CTC-based|||


# End-to-End Spoken Language Understanding (음성 언어 이해)
<pre>
<code>
End-to-End Spoken Language Understanding (SLU)란 음성을 입력으로 받아 direct로 감정(emotion)/의도(intent,slot) 등을 결과로 출력하거나 
Question-Answering (QA) 등의 문제를 푸는 task이다. 
</code>
</pre>

|year|conference|research organization|title|model|link|code|
|--|--|--|------|---|--|--|
|2019|Interspeech|MILA,CIFAR|Speech Model Pre-training for End-to-End Spoken Language Understanding||[paper](https://arxiv.org/pdf/1904.03670)|[code(official)](https://github.com/lorenlugosch/end-to-end-SLU)|


<br>

# Self-Supervised(or Semi-Supervised) Learning for Speech 
<pre>
<code>
Self-Supervised(or Semi-Supervised) Learning 이란 Yann Lecun이 강조했을 만큼 현재 2020년 현재 딥러닝에서 가장 핫 한 주제중 하나이며, 
Label되지 않은 방대한 data를 self-supervised (or semi-supervised) 방법으로 학습하여 입력으로부터 더 좋은 Representation을 찾는 방법이다. 
이렇게 사전 학습(pre-training)된 네트워크는 음성 인식 등 다른 task를 위해 task-specific 하게 미세 조정 (fine-tuning)하여 
scratch 부터 학습한 네트워크보다 더욱 높은 성능을 낼 수 있게 도와준다. 
사전 학습 방법은 AutoEncoder 부터 BERT 까지 다양한 방법으로 기존에 존재했으나 음성에 맞는 방식으로 연구된 논문들이 최근에 제시되어 왔으며, 
높은 성능을 자랑한다 .
</code>
</pre>

|year|conference|research organization|title|link|code|
|--|--|--|------|--|--|
|2019|-|Facebook AI Research (FAIR)|wav2vec: Unsupervised Pre-training for Speech Recognition|[paper](https://arxiv.org/pdf/1904.05862)|[code(official code)](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec)|
|2019|ICLR|Facebook AI Research (FAIR)|vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations|[paper](https://arxiv.org/pdf/1910.05453)|[code(official code)](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec)|
|2020|-|Facebook AI Research (FAIR)|wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations|[paper](https://arxiv.org/pdf/2006.11477)|[code(official code)](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec)|
|2019|-|Deepmind|Learning robust and multilingual speech representations|[paper](https://arxiv.org/pdf/2001.11128)||
|-|-|Deepmind|Self-Supervised Representations Improve End-to-End Speech Translation|[paper](https://arxiv.org/pdf/1508.01211)||
|-|-|Deepmind|Unsupervised Pretraining Transfers Well Across Languages|||
|-|-|Deepmind|Learning Problem-agnostic Speech Representations from Multiple Self-supervised Tasks|||
|-|-|Deepmind|Learning robust and multilingual speech representations|||
|-|-|Deepmind|Problem-Agnostic Speech Embeddings for Multi-Speaker Text-to-Speech with SampleRNN|||
|2020|-|MIT CSAIL|SEMI-SUPERVISED SPEECH-LANGUAGE JOINT PRE- TRAINING FOR SPOKEN LANGUAGE UNDERSTANDING|[paper](https://arxiv.org/pdf/2010.02295)||
