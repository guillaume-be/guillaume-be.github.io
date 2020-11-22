---
layout: post
title: "Accelerating text generation with Rust"
author: "Guillaume"
---

Over the past few months, text generation capabilities using Transformer-based models has been democratized by Hugging Face's Transformers [[1]](#transformers) library. A broad range of models and applications have been made available, including:
- Summarization models fine-tuned on the CNN-DailyMail [[2]](#cnndailymail) or XSUM [[3]](#xsum) datasets, including for example BART [[4]](#bart) or T5 [[5]](#t5)
- Translation, with the availability of more than a thousand models trained by the Opus-MT team from Language Technology at the University of Helsinki [[6]](#opusmt)
- Text generation, generating tokens from a prompt text, including OpenAI's GPT2 model [[7]](#gpt2)

These models offer state-of-the-art performance and can be set-up with just a few lines of code using ready-to-use pipelines in the Transformers library. This allowed a wide spread of the recent advances in the field of NLP to the industry where they can be effectively used to solve business-driven use cases.

To illustrate both translation and summarization capabilities, we'll use this news article from  WikiNews shared under  CC BY 2.5 license [[8]](#wikinews): 

> In findings published Tuesday in Cornell University's arXiv by a team of scientists from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's habitable zone — not too hot and not too cold for liquid water to exist.
The Montreal team, led by Björn Benneke, used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water, weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software and confirmed their conclusion.
This was not the first time scientists have found signs of water on an exoplanet, but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth.
"This is the first potentially habitable planet where the temperature is right and where we now know there is water," said UCL astronomer Angelos Tsiaras. "It's the best candidate for habitability right now."
"It's a good sign", said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors. "Overall," he continued, "the presence of water in its atmosphere certainly improves the prospect of K2-18b being a potentially habitable planet, but further observations will be required to say for sure."
K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year on K2-18b lasts 33 Earth days.
According to The Guardian, astronomers were optimistic that NASA's James Webb space telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more about exoplanets like K2-18b.

#### Summarization
This article can be summarized calling the following snippet from the Transformer's Python library [[1]](#transformers), defaulting to a BART model trained on the CNDailyMal dataset:
```python
from transformers import pipeline
summarization_pipeline = pipeline("summarization")

summarization_pipeline(input_article)
```
returning:
>
### ToDo: add resulting summarization

#### Translation
Similarly, a translation model can be easily created to translate a sentence of this document to Spanish:
```python
from transformers import pipeline
translation_pipeline = pipeline("translation_en_to_es")

translation_pipeline("They found that certain wavelengths of light, which are usually \
 absorbed by water, weakened when the planet was in the way, indicating not only \
 does K2-18b have an atmosphere, but the atmosphere contains water in vapour form.")
```
returning: 
>
>### ToDo: add resulting translation

#### Text generation
Text can be generated using a text generation pipeline:
```python
from transformers import pipeline
text_generator = pipeline("text-generation")

print(text_generator("As far as I am concerned, I will", 
    max_length=32, 
    do_sample=False))
```

This article will briefly describe the architecture of such models before diving in a comparison of the baseline Python Transformers library with a proposed Rust-based implementation: rust-bert [[9]](#rustbert)

# Overview of Summarization and Translation models

Translation and summarization both rely on a similar architecture, although the model weights natural vary from application to application. They are essentially made of:
1. A pre-processing pipeline mostly comprising of a tokenizer (such as Byte Pair Encoding or SentencePiece/Unigram-based) and an encoder (mapping individual tokens to a vocabulary index and other optional inputs (such as position indices)).

2. A transformer-based model, based on an encoder-decoder architecture. If you are not familiar with Transformer-based encoder-decoder architecture, I highly recommend the blog post "The Illustrated Transformer" [[10]](#illustratedtransformer). The encoder is comprised of a stack of self-attention and fully connected layers and encodes the input sequence (i.e. text to be translated or summarized) into a latent space. The decoder is made of a similar stack of self-attention layers completed with cross-attention to the encoder hidden states, allowing to leverage the representations generated during encoding. The decoder takes as an input the output sequence generated so far and the encoder output to generate the next tokens. The decoder therefore generates output tokens one at a time.

3. A generation routine, which in its simplest form will keep calling the transformer-based models to generate tokens until the sequence is completed (output of an `End Of Sequence` token). Note that the encoder only needs to be run once in this iterative process: its output is cached and re-used at each decoder generation step. In practice, more advanced algorithms are used to improve the quality of the generation, including beam search, sampling, length and repetition penalties. These methods are summarized in an excellent article from Hugging Face [[11]](#generation). Careful design of the decoder allows to not only cache the encoder states, but also parts of the keys and values in the decoder to avoid unnecessary re-calculation and speed-up the decoding process.

This iterative process is illustrated at a high level in the figure below:
### ToDo: add the generation process illustration

This process (and in the special case of BART and Marian - the model architecture itself) is identical between translation and summarization. Only the tokenization process and the model parameters differ between the two applications, showing the high versatility of this system. 

One may also note the complexity of the generation routine, involving a significant number of operations beyond the model forward pass, helping to improve the quality of the generated text. However, these improvements do not come for free and incur additional computational cost in both the model forward pass (beam search increases the effective batch size) and in post-processing operations.

The Python Transformer's library already leverages Rust-based tokenizers for all its ready-to-use pipelines, therefore accelerating the preprocessing part of the system. Some benchmarks on question answering [[9]](#rustbert) indicate that the preprocessing can amount to 20% to 30% of the processing time for simple pipelines. The post-processing, also involving operations that go beyond tensor operations, is however implemented in Python in the Transformer's library. The rest of this article assesses the impact of a high-performance implementation of the entire system in Rust (therefore covering the post-processing pipeline) using the rust-bert library [[9]](#rustbert).

# A brief introduction to rust-bert

Rust-bert is essentially a Rust-native port of Hugging Face's Transformers' library [[1]](#transformers). Leveraging the `rust_tokenizers` [[12]](#rusttokenizers) library for preprocessing, it proposes implementations for state-of-the-art transformers-based models and ready-to-use pipelines. De-noising auto-encoder (BERT, Electra), autoregressive (XLNet, GPT, GPT2) and encoder-decoder models (BART, T5) have been implemented with pre-trained set of weights available on Hugging Face's model hub [[13]](#modelhub). Any Pytorch model trained on the Transformers's library can be converted to a C-array format and used by the `rust-bert` library.

These models can be used in ready-to-use pipelines, including:
- classification (e.g. sentiment analysis)
- token classification (e.g. named entities recognition)
- extractive question answering
- zero-shot classification, using a natural language inference classifier
- text generation
- conversational model
- translation
- summarization

The last two pipelines allow for a side-by-side comparison of the Python implementation (with Rust tokenizers) and the end-to-end Rust version. The two pipelines mentioned above can also be instantiated in a few lines of code:

#### Summarization
```rust
use rust_bert::pipelines::summarization::SummarizationModel;
let summarization_model = SummarizationModel::new(Default::default())?;

summarization_model.summarize(&input);
```

#### Translation
```rust
use rust_bert::pipelines::translation::{Language, TranslationConfig, TranslationModel};
let translation_config =
    TranslationConfig::new(Language::EnglishToSpanish, Device::cuda_if_available());
let model = TranslationModel::new(translation_config)?;

model.translate(&["They found that certain wavelengths of light, which are usually 
 absorbed by water, weakened when the planet was in the way, indicating not only 
 does K2-18b have an atmosphere, but the atmosphere contains water in vapour form."]);
```
ToDo:
- schema of the generation process (preprocessing, encoder hidden states, generation loop with potential improvement options, decoder input, caching...)
- Mention the algorithm is identical, only different models/weights are used for translation and summarization
- note on the complexity of the post-processing operations
- (brief) introduction to rust-bert
    - short abstract about what the library is about
    - link to paper describing the system and to github
    - example of how the summarization and translation would be called in Rust
- Goal of article: how does Rust high performance impact the generation process

- Benchmark text generation techniques
    - description of experimental setup (hardware and library versions)
    - description of inputs for each task + benchmark: 
        1. describe the input
        2. describe the options (mention that the defaults may be different)
        2. show the numbers
        3. comment
    - show it for text generation


- Next steps:
    - synergies with model optimization techniques (distillation, quantization, pruning)
    - synergies with optimized runtimes
    

## References
- <a name="transformers"></a>[1] [Transformers: State-of-the-Art Natural Language Processing
](https://www.aclweb.org/anthology/2020.emnlp-demos.6/), Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, Alexander Rush. 
- <a name="cnndailymail"></a>[2] [Get To The Point: Summarization with Pointer-Generator Networks](http://arxiv.org/abs/1704.04368), Abigail See, Peter J. Liu, Christopher D. Manning
- <a name="xsum"></a>[3] [Don’t Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://www.aclweb.org/anthology/D18-1206/), Shashi Narayan, Shay B. Cohen, Mirella Lapata
- <a name="bart"></a>[4] [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://www.aclweb.org/anthology/2020.acl-main.703/), Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, Luke Zettlemoyer
- <a name="t5"></a>[5] [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683), Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu
- <a name="opusmt"></a>[6] <https://github.com/Helsinki-NLP/Opus-MT>
- <a name="gpt2"></a>[7] [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf), Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever
- <a name="wikinews"></a>[8] [Astronomers find water vapour in atmosphere of exoplanet K2-18b](https://en.wikinews.org/wiki/Astronomers_find_water_vapour_in_atmosphere_of_exoplanet_K2-18b), WikiNews
- <a name="rustbert"></a>[9] [End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4/), Becquin, Guillaume
- <a name="illustratedtransformer"></a>[10] [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/), Alammar, Jay
- <a name="generation"></a>[11] [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate), von Platen, Patrick
- <a name="rusttokenizers"></a>[12] [rust_tokenizers](https://github.com/guillaume-be/rust-tokenizers)
- <a name="modelhub"></a>[13] [Hugging Face model hub](https://huggingface.co/models?filter=rust)