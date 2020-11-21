---
layout: post
title: "Accelerating text generation with Rust"
author: "Guillaume"
---

Over the past few months, text generation capabilities using Transformer-based models has been democratized by Hugging Face's Transformers [[1]](#transformers) library. A broad range of models and applications have been made available, including:
- Summarization models fine-tuned on the CNN-DailyMail [[2]](#cnndailymail) or XSUM [[3]](#xsum) datasets, including for example BART [[4]](#bart) or T5 [[5]](#t5)
- Translation, with the availability of more than a thousand models trained by the Opus-MT team from Language Technology at the University of Helsinki [[6]](#opusmt)

### ToDo: Provide an example to illustrate the model capabilities (with code in Python)
- summarization
- translation


These models offer state-of-the-art performance and can be set-up with just a few lines of code using ready-to-use pipelines in the Transformers library. This allowed a wide spread of the recent advances in the field of NLP to the industry where they can be effectively used to solve business-driven use cases.

This article will briefly describe the architecture of such models before diving in a comparison of the baseline Python Transformers library with a proposed Rust-based implementation: rust-bert [[7]](#rustbert)

# Overview of Summarization and Translation models

Translation and summarization both rely on a similar architecture, although the model weights natural vary from application to application. They are essentially made of:
1. A pre-processing pipeline mostly comprising of a tokenizer (such as Byte Pair Encoding or SentencePiece/Unigram-based) and an encoder (mapping individual tokens to a vocabulary index and other optional inputs (such as position indices)).

2. A transformer-based model, based on an encoder-decoder architecture. If you are not familiar with Transformer-based encoder-decoder architecture, I highly recommend the blog post "The Illustrated Transformer" [[8]](#illustratedtransformer). The encoder is comprised of a stack of self-attention and fully connected layers and encode the input sequence (i.e. text to be translated or summarized) into tensor representations. The decoder is made of a similar stack of self-attention layers completed with cross-attention to the encoder hidden states, allowing to leverage the representations generated during encoder. The decoder takes as an input the output sequence generated so far and the encoder output to generate the next tokens. The decoder therefore generates output tokens one at a time.

3. A generation routine, which in its simplest form will keep calling the transformer-based models to generate tokens until the sequence is completed (eneration of an `End Of Sequence` token). Note that the encoder only needs to be run once in this iterative process: its output is cached and re-used at each decoder generation step. In practice, more advanced algorithms are used to improve the quality of the generation, including beam search, sampling, length and repetition penalties. These methods are summarized in an excellent article from Hugging Face[[9]](#generation)


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
        2. show the numbers
        3. comment
    


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
- <a name="rustbert"></a>[7] [End-to-end NLP Pipelines in Rust](https://www.aclweb.org/anthology/2020.nlposs-1.4/), Becquin, Guillaume
- <a name="illustratedtransformer"></a>[8] [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/), Alammar, Jay
- <a name="generation"></a>[9] [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate), von Platen, Patrick