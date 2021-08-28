---
layout: post
title: "Byte Pair Encoding and Data Structures"
author: "Guillaume"
---

Tokenization of input strings into a sequence of word or sub-tokens is a central concept for modern Natural Language Processing techniques (NLP). This article proposes an implementation of a classic tokenization algorithm: Byte Pair Encoding (BPE) [[1]](#bpe). Multiple algorithms and their implementation in Rust will illustrate how the choice of data structures impact the performance of a real-world NLP component.

# Introduction

Straight-forward word-based segmentation has been a de-facto pre-processing standard step for years. When combined with n-grams computation, one-hot-encoding, frequency weighting (such as term frequency - inverse document frequency, tf-idf [[2]](#tf-idf)) or dense word embeddings (such as GloVe [[3]](#glove)). Word-based tokenization simplicity needs to be balanced by a few important drawbacks: because the total number of words in most languages' vocabulary is very large (including very large words), this approach will either result in very large input dimensionality (very broad - although sparse - frequency matrices or very long embeddings matrices) or will rely on converting tokens that were pruned from the vocabulary by an "unknown" token. Even when including the entire valid vocabulary for the generation of these encoding matrices, common input corruption such as typos will result in out-of-vocabulary inputs. Improved techniques, such as FastText embeddings [[4]](#fasttext) allow using word-based inputs augmented with sub-word information, handling high morphology languages and rare words better.

At the other end of the spectrum, character-based tokenization usually allows for a very small encoding matrix at the cost of a lower representation power: by splitting the input into a sequence of characters or unicode points, one can ensure a full conversion of the input without loss of information ("unknown" tokens). The cost is a low semantic information content of such characters, and the need to rely on powerful downstream models to generate meaningful features. This has been successfully illustrated in approaches such as CANINE [[5]](#canine).

Halfway between word-based tokenization and character input, sub-word tokenization aims at combining the benefits of both approaches. Sub-word tokenization works by splitting rare words into sub-word components instead of ignoring them or having a seldom-used dedicated representation for them. Ideally, these sub-word representation result in a segmentation that is also semantically or syntactically meaningful, for example splitting pre- or suffixes from words (for example, `unexpected` would become `[un, ##expected]`). This is in practice not always the case and depends on the sub-word segmentation algorithm and the corpus used for training. The resulting vocabulary size can be an order of magnitude smaller than for word-based segmentation while common words will be represented as a single entry - allowing to generate semantically-rich word embeddings for the downstream model.

These models usually have two algorithmic components: 
- a training component, where a vocabulary and rules to merge/split words is learned using an unsupervised algorithm.
- a prediction component, where these rules are used to split an input sentence or documents into a sequence of words/sub-tokens.

While proper algorithmic design of the training phase is important to allow scaling to larger corpora, this is a one-off cost. This article focuses on the design and implementation of the prediction phase which is critical to limit the latency and operation cost of models when they are deployed and generate production. Following a previous article illustrating the [SentencePiece unigram tokenization model](../2020-05-30/sentence_piece), this article will focus on another ubiquitous algorithm: Byte Pair Encoding (BPE).

# 1. The Byte Pair Encoding (BPE) tokenizer

In contrast with SentencePiece statistical unigram model, BPE is a morphological tokenizer that merges adjacent byte pairs based on their frequency in a training corpus. Based on a compression algorithm with the same name, BPE has been adapted to sub-word tokenization that can be thought of as a clustering algorithm [[6]](#bpe-lecture). A starting sequence of individual characters will be aggregated bottom-up based on frequencies learned during a training phase. When no further aggregation is possible, end the process and return the sub-words generated. 

Once can note that no unknown tokens can occur as an output of this process as long as all individual characters are present in the vocabulary (in the worst case, the sequence of individual characters will be returned). With a sufficiently large vocabulary, the most common words will be fully aggregated and the entire sequence of input characters will be returned as a single word token. Rare words will be returned as a list of sub-tokens that can no longer be aggregated.

The following example illustrates how a merges vocabulary (mapping of symbols that may be aggregated with their frequency) is used to tokenize 2 words. Note that in practice, a `</w>` symbol is appended at the end of the initial word character decomposition and included in the merges vocabulary to differentiate end of words symbols.
```json
{
  "e r": 25,
  "h e": 12,
  "l l": 11,
  "l o": 10,
  "he ll": 4,
  "lo w": 3,
  "hell o": 2 
}
```
"lower" would follow the following aggregation steps: <br>
`[l, o, w, e, r] -> [l, o, w, er] -> [lo, w, er] -> [low, er]`

While "hello" would be tokenized as: <br>
`[h, e, l, l, o] -> [he, l, l, o] -> [he, ll, o] -> [hell, o] -> [hello]`

## 1.a Training

For computational efficiency, BPE training relies on a pre-tokenizer (e.g. whitespace tokenizer) in order to generate a dictionary with word frequencies (the original BPE algorithm does not allow cross-word merges). This word counter is used to initialize a counter of adjacent "bytes" (symbols) pairs that may be merged. This "symbol pairs" counter is used to create a new symbol in the dictionary, update the symbol pairs counts and repeat until the target vocabulary size is reached. As such, BPE is an  algorithm that grows its vocabulary at each iteration (in contrast with SentencePiece unigram's model that prunes a large vocabulary at each iteration).

The entire training algorithm is rather straight-forward and given below (reproduced from [[1]](#bpe)).

{% include pseudocode.html id="1" code="
\begin{algorithm}
\caption{BPE Train}
\begin{algorithmic}
\PROCEDURE{UpdateMerges}{$vocab$}
    \STATE $merges = $\CALL{InitializeMerges}{}
    \FOR{$(word, freq)$ IN $Vocab$}
        \STATE $symbols = $ \CALL{SplitWhitespace}{$word$}
        \FOR{$i = 0$ \TO $len(symbols)$}
            \STATE $merges[symbols[i], symbols[i+1]] += freq$
        \ENDFOR
    \ENDFOR
    \RETURN $merges$
\ENDPROCEDURE
\PROCEDURE{MergeVocab}{$vocab_in, pair$}
    \STATE $vocab_out = $\CALL{InitializeVocab}{}
    \STATE $pattern = $ \CALL{Join}{pair}
    \FOR{$(w_{in}, _)$ IN $vocab_in$}
        \STATE $w_{out} = $ \CALL{Replace}{ $w_{in}$, $pair_{0} \sqcup pair_{1}$, $pair_{0} pair_{1}$}
        \STATE $vocab_{out}[w_{out}] = vocab_{in}[w_{in}]$
    \ENDFOR
    \RETURN $vocab_{out}$
\ENDPROCEDURE
\PROCEDURE{Train}{$corpus$}
    \STATE vocab = \CALL{CountWords}{$corpus$}
    \FOR{$i = 0$ \TO $N_{merges}$}
        \STATE $merges = $ \CALL{UpdateMerges}{$vocab$}
        \STATE $most\:common\:pair = $ \CALL{MostCommon}{$merges$}
        \STATE $vocab = $ \CALL{MergeVocab}{$vocab$, $most\:common\:pair$}
    \ENDFOR
    \RETURN $merges$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
" %}


## References
- <a name="bpe"></a>[1] [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), Rico Sennrich, Barry Haddow, Alexandra Birch, 2015 
- <a name="tf-idf"></a>[2] [term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf–idf), Wikipedia foundation. 
- <a name="glove"></a>[3] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf), Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
- <a name="fasttext"></a>[4] [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf), Bojanowski, Grave, Joulin,  Mikolov. 2016.
- <a name="canine"></a>[5] [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874), Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting, 2021.
- <a name="bpe-lecture"></a>[6] [Natural Language Processing
with Deep Learning, CS224N/Ling284, lecture 12](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture12-subwords.pdf), Christopher Manning, Stanford.