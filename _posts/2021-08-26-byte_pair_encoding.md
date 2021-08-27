---
layout: post
title: "Byte Pair Encoding and Data Structures"
author: "Guillaume"
---

Tokenization of input strings into a sequence of word or sub-tokens is a central concept for modern Natural Language Processing techniques (NLP). This article proposes an implementation of a classic tokenization algorithm: Byte Pair Encoding (BPE). Multiple algorithms and their implementation in Rust will illustrate how the choice of data structures impact the performance of a real-world NLP component.

# Introduction

Straight-forward word-based segmentation has been a de-facto pre-processing standard step for years. When combined with n-grams computation, one-hot-encoding, frequency weighting (such as term frequency - inverse document frequency, tf-idf [[1]](#tf-idf)) or dense word embeddings (such as GloVe [[2]](#glove)). Word-based tokenization simplicity needs to be balanced by a few important drawbacks: because the total number of words in most languages' vocabulary is very large (including very large words), this approach will either result in very large input dimensionality (very broad - although sparse - frequency matrices or very long embeddings matrices) or will rely on converting tokens that were pruned from the vocabulary by an "unknown" token. Even when including the entire valid vocabulary for the generation of these encoding matrices, common input corruption such as typos will result in out-of-vocabulary inputs.

At the other end of the spectrum, character-based tokenization usually allows for a very small encoding matrix at the cost of a lower representation power: by splitting the input into a sequence of characters or unicode points, one can ensure a full conversion of the input without loss of information ("unknown" tokens). The cost is a low semantic information content of such characters, and the need to rely on powerful downstream models to generate meaningful features. This has been successfully illustrated in approaches such as CANINE [[3]](#canine).

Halfway between word-based tokenization and character input, sub-word tokenization aims at combining the benefits of both approaches. Sub-word tokenization works by splitting rare words into sub-word components instead of ignoring them or having a seldom-used dedicated representation for them. Ideally, these sub-word representation result in a segmentation that is also semantically or syntactically meaningful, for example splitting pre- or suffixes from words (for example, `unexpected` would become `[un, ##expected]`). This is in practice not always the case and depends on the sub-word segmentation algorithm and the corpus used for training. The resulting vocabulary size can be an order of magnitude smaller than for word-based segmentation while common words will be represented as a single entry - allowing to generate semantically-rich word embeddings for the downstream model.

These models usually have two algorithmic components: 
- a training component, where a vocabulary and rules to merge/split words is learned using an unsupervised algorithm.
- a prediction component, where these rules are used to split an input sentence or documents into a sequence of words/sub-tokens.

While proper algorithmic design of the training phase is important to allow scaling to larger corpora, this is a one-off cost. This article focuses on the design and implementation of the prediction phase which is critical to limit the latency and operation cost of models when they are deployed and generate production. Following a previous article illustrating the [SentencePiece unigram tokenization model](../2020-05-30/sentence_piece), this article will focus on another ubiquitous algorithm: Byte Pair Encoding (BPE).

# 1. The Byte Pair Encoding tokenizer





## References
- <a name="tf-idf"></a>[1] [term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf–idf), Wikipedia foundation. 
- <a name="glove"></a>[2] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf), Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
- <a name="canine"></a>[2] [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874), Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting, 2021.
