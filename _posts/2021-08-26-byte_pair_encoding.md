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

## Tokenization (prediction)

Even though the BPE model needs to be trained before it can be used, we will first describe how it is used to tokenize an input sequence at prediction time in this article. This will help build an intuition on how the algorithm decomposes words into sub-tokens.

The tokenization algorithm is as follows:

<a name="bpe-tokenization-naive"></a>
{% include pseudocode.html id="0" code="
\begin{algorithm}
\caption{BPE Tokenize}
\begin{algorithmic}
\PROCEDURE{FindBestPair}{$symbols$, $merges$}
    \STATE ($best\:pair$, $best\:score$) = ($null$, 0)
    \FOR{$i = 0$ \TO $len(symbols) - 1$}
        \STATE $score$ = \CALL{Get}{$merges$, $(symbols[i], symbols[i+1])$}
        \IF{$score$ > $best\:score$}
            \STATE ($best\:pair$, $best\:score$) = ($(symbols[i], symbols[i+1])$, score)
        \ENDIF
    \ENDFOR    
    \RETURN ($best\:pair$, $best\:score$)
\ENDPROCEDURE
\STATE
\PROCEDURE{MergeBestPair}{$symbols$, $best\:pair$}
    \FOR{$i = 0$ \TO $len(symbols) - 1$}
        \IF{$(symbols[i], symbols[i+1])$ = $best\:pair$}
            \STATE \CALL{Merge}{$symbols[i], symbols[i+1]$}
        \ENDIF
    \ENDFOR    
    \RETURN $symbols$
\ENDPROCEDURE
\STATE
\PROCEDURE{Tokenize}{$text$, $merges$}
    \STATE $symbols = $ \CALL{InitializeSymbols}{$text$} \COMMENT{Pre-populate symbols with individual characters and </w> token}
    \WHILE {$true$}
        \STATE $(best\:pair, best\:score) = $ \CALL{FindBestPair}{$symbols$, $merges$}
        \IF{$best\:score$ is $null$}
            \BREAK \COMMENT{Symbols can no longer be merged}
        \ENDIF
        \STATE $symbols$ = \CALL{MergeBestPair}{$symbols$, $best\:pair$}
    \ENDWHILE
    \RETURN $symbols$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
" %}

`merges` is a learned mapping of symbol pairs to score that is learned during a training phase (see next section). The following example illustrates how a merges vocabulary (mapping of symbols that may be aggregated with their frequency) is used to tokenize 2 words. Note that in practice, a `</w>` symbol is appended at the end of the initial word character decomposition and included in the merges vocabulary to differentiate end of words symbols.
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
- "lower" would follow the following aggregation steps: <br>
`[l, o, w, e, r] -> [l, o, w, er] -> [lo, w, er] -> [low, er]` (`"low er"` not in merges vocabulary, causing an early stopping of the tokenization procedure at line _26_)

- "hello" would be tokenized as: <br>
`[h, e, l, l, o] -> [he, l, l, o] -> [he, ll, o] -> [hell, o] -> [hello]`

## Training

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
\STATE
\PROCEDURE{MergeVocab}{$vocab_{in}, pair$}
    \STATE $vocab_{out} = $\CALL{InitializeVocab}{}
    \STATE $pattern = $ \CALL{Join}{pair}
    \FOR{$(w_{in}, _)$ IN $vocab_{in}$}
        \STATE $w_{out} = $ \CALL{Replace}{ $w_{in}$, $pair_{0} \sqcup pair_{1}$, $pair_{0} pair_{1}$}
        \STATE $vocab_{out}[w_{out}] = vocab_{in}[w_{in}]$
    \ENDFOR
    \RETURN $vocab_{out}$
\ENDPROCEDURE
\STATE
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

The algorithm above can be improved as described in [[1]](#bpe) by updating data structures instead of re-computing the "merges" from scratch at each iteration. This article will however focus on the tokenization procedure called at prediction time.

# 2. Rust implementation(s) of the BPE algorithm

The rest of this article consists of a walk-through a number of working implementations of the BPE tokenization algorithm in Rust. Starting from a "naive" implementation approach, improvements will be made to highlight pros and cons of some common data structures within the scope of BPE tokenization.

4 algorithms for the BPE tokenization will be presented:
- [Naive implementation](#naive)
- [Naive implementation with pre-splitting](#naive-pre-split)
- [Priority Queue + Binary Search Tree implementation](#pq-bst)
- Priority Queue + Linked List implementation

## a. <a name="naive"></a>Naive implementation

Let's begin with a direct implementation of [[algorithm 1]](#bpe-tokenization-naive), which consists in 2 main procedures:
1. Find the best merge
2. Apply the best merge

Until no valid merge can be found. Let's examine the algorithm complexity, where _N_ represents the number of characters of an input text. Assuming a lookup in the merges mapping can be done in constant time (a fair assumption for a standard hashmap implementation), the `FindBestPair` procedure has a $O(N)$ complexity (_symbols_ is of length _N_ for the first iteration). Similarly, the `MergeBestPair` has a $O(N)$ complexity if the `Merge` operation can be done in constant time. The main `Tokenize` procedure will iterate until no further merge can be found, which will result in _N-1_ merges in the "worst" case (if the entire word exists in the vocabulary). This results in a combined complexity of $O(N^{2})$. This motivates the pre-tokenization (e.g. whitespace splitting) step that usually precedes BPE tokenization, so that _N_ is the typical length of a word rather than a full sentence/document.

Let's look at a Rust implementation of this algorithm. The following code has been edited to focus on the critical parts of the algorithm. The full working version can be found at [[7]](#bpe-code). Let's start by defining `Symbols`, the data structure representing a sub-token:

```rust
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Symbol {
    pub start_byte: usize,
    pub end_byte: usize,
}
```

A symbol contains information related to the start and end byte positions of the sub-token. This is lighter than working with string slices (or significantly faster than string clones) and convenient for the operations we expect on `Symbols` (merges and lookups). We will not manipulate `Symbols` on their own, but rather work on a collection of `Symbols`. The naive implementation, looping over the list of symbols indicates a Rust `Vec` may be an appropriate structure to use:

```rust
pub struct SymbolArray {
    pub symbols: Vec<Symbol>,
}
```

The symbols are initialized from the characters of the input text (see algorithm 1, line _21_). We will create a method `from_text` that populates the initial `SymbolArray` from a string slice. Note that we look up character byte indices so that we handle characters that span over multiple UTF-8 bytes correctly.

We also implement a method for finding the best pair to merge (given a tokenizer/merge dictionary) that will return an optional position of the best pair in the `SymbolArray`. When this method return `None`, we know that the array can no longer be merged. Finally, we implement directly a method to merge a pair of symbols mutating the SymbolArray inplace. This method will take the best pair position as an input to directly insert the merged pair and remove the two parents.

```rust
impl SymbolArray {
    pub fn from_text(input_text: &str) -> Self {
        let mut symbols = Vec::new();
        for (character_start, character) in input_text.char_indices() {
            symbols.push(Symbol {
                start_byte: character_start,
                end_byte: character_start + character.len_utf8(),
            });
        }
        Self { symbols }
    }

    pub fn find_best_merge<T>(&self, input_text: &str, tokenizer: &T) -> Option<usize>
    where
        T: BpeTokenizer,
    {
        self.symbols
            .iter()
            .tuple_windows::<(&Symbol, &Symbol)>()
            .enumerate()
            .filter_map(|(pos, (first, second))| {
                tokenizer
                    .get_merge_score(first, second, input_text)
                    .map(|rank| (pos, rank))
            })
            .min_by_key(|(_, rank)| *rank)
            .map(|(pos, _)| pos)
    }

    pub fn merge_symbols(&mut self, best_pair_index: usize) -> Symbol {
        let new_symbol = Symbol {
            start_byte: self.symbols[best_pair_index].start_byte,
            end_byte: self.symbols[best_pair_index + 1].end_byte,
        };
        self.symbols.remove(best_pair_index + 1);
        self.symbols.remove(best_pair_index);
        self.symbols.insert(best_pair_index, new_symbol);
        new_symbol
    }
}
```

We still need to implement the actual tokenizer and `tokenize` procedure. This will pre-process the input text (replaces whitespaces by the corresponding `▁` encoding symbol in the pre-trained vocabulary), pre-populate a `SymbolArray` and identify/merge best symbol pairs until no further merge is possible. It then returns string slices with the sub-tokens. At no point the tokenizer creates a copy of the string input: it solely relies on byte positions and string slices.

```rust
pub struct NaiveBpeTokenizer {
    merges_vocab: MergesVocab,
}

impl BpeTokenizer for NaiveBpeTokenizer {

    fn tokenize<'a>(&self, input_text: &'a str) -> Vec<&'a str> {
        let (text, byte_mapping) = self.pre_process_text(input_text, '▁');

        let mut symbols = SymbolArray::from_text(text.as_str());
        while let Some(best_pair_index) = symbols.find_best_merge(text.as_str(), self) {
            symbols.merge_symbols(best_pair_index);
        }
        let mut output = Vec::new();
        for symbol in symbols {
            output.push(
                &input_text[byte_mapping[&symbol.start_byte]..byte_mapping[&symbol.end_byte]],
            );
        }
        output
    }
}
```

As mentioned previously, this algorithm has a $O(N^{2})$ complexity where N represents the number of characters in the input. This means that this algorithm will be unable to process long input text of the size of a sentence, paragraph or document. A common work-around consists in limiting the size of the input passed to the BPE tokenizer, for example by applying a whitespace splitting pre-tokenizer.

## b. <a name="naive-pre-split"></a>Pre-splitting naive implementation

The previous implementation can easily be extended to include a pre-tokenization step. This is actually the standard BPE implementation in several widely used packages, such as subword-nmt (Python) [[8]](#subword-nmt) or fastBPE (C++) [[9]](#fastbpe). By pre-tokenizing the sequence (for example whitespace splitting), one can effectively limit the average size of the inputs passed for BPE tokenization. A whitespace splitting will pass single words for processing. For most languages, the expected size of a word _M_ is much smaller than the number of characters in an average sentence or document _N_ (although after living in Germany for 10 years, the author realizes this hypothesis may be optimistic at times). The complexity of the tokenization for a word is $O(M^{2})$. Splitting the sequence into words using a simple whitespace/punctuation rule has a $O(N)$ complexity, resulting in a number of words that is at most _N_, meaning the algorithm complexity is $O(N) + O(N*M^{2}) = O(N)$ if $M \ll N$.

To implement this tokenizer, the `Tokenizer` has an additional method for whitespace/punctuation tokenization. Punctuation are returned as a single character word, and other words contain the leading whitespace character if applicable. The method returns a vector of string slices:

```rust
impl NaivePreSplitBpeTokenizer {
    fn split_whitespace_punctuation<'a>(
        &self,
        input_string: &'a str,
        whitespace_token: char,
    ) -> Vec<&'a str> {
        let mut output: Vec<&str> = Vec::new();
        let mut start: usize = 0;

        for (c_pos, c) in input_string.char_indices() {
            if c == whitespace_token {
                if start < c_pos {
                    output.push(&input_string[start..c_pos]);
                }
                start = c_pos;
            } else if c.is_ascii_punctuation() {
                if start < c_pos {
                    output.push(&input_string[start..c_pos]);
                }
                output.push(&input_string[c_pos..c_pos + c.len_utf8()]);
                start = c_pos + c.len_utf8();
            }
        }
        if start < input_string.len() {
            output.push(&input_string[start..]);
        }
        output
    }
}
```

The previous `tokenize` method can be modified to include the pre-tokenization step. This is essentially identical to the previous method, with the addition of the whitespace splitting and additional logic to keep track of the correct character offsets to return the sub-tokens:
```rust
impl BpeTokenizer for NaivePreSplitBpeTokenizer {
    fn get_merges_vocab(&self) -> &MergesVocab {
        &self.merges_vocab
    }

    fn tokenize<'a>(&self, input_text: &'a str) -> Vec<&'a str> {
        let whitespace_token = '▁';

        let (text, byte_mapping) = self.pre_process_text(input_text, whitespace_token);
        let split_texts = self.split_whitespace_punctuation(text.as_str(), whitespace_token);

        let mut output = Vec::new();
        let mut offset = 0;
        for split_text in split_texts {
            let mut symbols = SymbolArray::from_text(split_text);
            while let Some(best_pair_index) = symbols.find_best_merge(split_text, self) {
                symbols.merge_symbols(best_pair_index);
            }
            for symbol in symbols.symbols {
                output.push(
                    &input_text[byte_mapping[&(offset + symbol.start_byte)]
                        ..byte_mapping[&(offset + symbol.end_byte)]],
                );
            }
            offset += split_text.len();
        }
        output
    }
}
```

This implementation brings the expected complexity from $O(N^{2})$ to $O(N)$ which is the best that can be done since the input needs to be scanned at least once to perform tokenization. This implementation, however, has the following drawbacks:
- A whitespace/punctuation tokenization step is required. This works well for most latin language, but may be problematic for example for some Asian languages not relying on whitespaces. This is a significant limitation and requires handling languages differently based on their morphology and use of word separators. 
- While faster, this algorithm is an approximation and prevents cross-word merges. It would for example be unable to merge words across punctuation symbols (such as "mother-in-law")

The following sections will present two additional implementation that will aim at reducing the computational cost of the naive algorithm without relying on a pre-tokenization step.

## c. <a name="pq-bst"></a>Priority Queue + Binary Search Tree implementation

The nave algorithm suffers from a worst case $O(N^{2})$ complexity. This is due to up to _N_ iterations of `FindBestPair` and `MergeBestPair`. Generally, an input may be made from up to _N-1_ valid pairs that will be merged, and the outer loop complexity can therefore not be optimized. 

The `MergeBestPair` complexity can be reduced to the time required to select the symbols to merge based on the best pair information if a single merge is performed at every iteration. The symbols can be compared and ordered based on their position - and a Binary Search Tree implementation of the _Symbols_ sequence allows finding elements in $O(\log(N))$ time instead of linear time for an array.

The `FindBestPair` can be optimized by noting that the entire symbol pair information does not need to be re-computed entirely at each iteration, but rather updated after every merge. Merging two symbols will have a local impact on the merge information: only the symbols preceding and succeeding the pair to be merged are impacted: we need to check if the preceding symbol and newly created merged symbol form a valid pair. Similarly, we need to check if the newly created merged symbol and next symbol form a valid pair. We still need a data structure to store the symbol pair information:
- _SymbolPairs_ can be compared and ordered based on their "cost": we need an ordered collection.
- At each iteration, the _SymbolPair_ with the minimum cost will be processed: we need a collection with an effective `extract_min` operation.
- At each iteration, up to 2 new _SymbolPairs_ will be inserted: we need a collection with an effective `insert` operation.

These requirements effectively describe a priority queue which will be used to build and maintain the set of _SymbolPairs_. Using a Min Heap allows performing both `extract_min` and `insert` operations in $O(\log(N))$ time.

By implementing the following data structures:
- Binary Search Tree for the _Symbols_
- Min Heap for the _SymbolPairs_,

A worst-case complexity of $O(N(\log(N) + \log(N)))$ = $O(N\log(N))$, significantly better than the initial $O(N^{2})$. The maintenance of a priority queue for the _SymbolPairs_ to process is mentioned in the SentencePiece article [[10]](#sentencepiece-paper) and is used in the optimized BPE implementation of the C++ SentencePiece library [[11]](#sentencepiece-bpe). _Algorithm 1_ (BPE tokenization) becomes:

{% include pseudocode.html id="2" code="
\begin{algorithm}
\caption{BPE Tokenize (Priority Queue)}
\begin{algorithmic}
\PROCEDURE{MaybeAddSymbolPair}{$symbol\:pairs$, $merges$, $symbol_{left}$, $symbol_{right}$}
    \STATE $score = $\CALL{Get}{$merges$, $(symbol_{left}, symbol_{right})$}
    \IF{$score \neq null$}
        \STATE \CALL{Insert}{$symbol\:pairs$, $(symbol_{left}, symbol_{right}, score)$}
    \ENDIF
\ENDPROCEDURE
\STATE
\PROCEDURE{MergeBestPair}{$symbols$, $best\:pair$}
    \STATE $new\:symbol = $ \CALL{Combine}{$best\:pair_{left}$, $best\:pair_{right}$}
    \STATE \CALL{Remove}{$symbols$, $best\:pair_{left}$}
    \STATE \CALL{Remove}{$symbols$, $best\:pair_{right}$}
    \STATE \CALL{Insert}{$symbols$, $new\:symbol$}
    \RETURN $new\:symbol$
\ENDPROCEDURE
\STATE
\PROCEDURE{Tokenize}{$text$, $merges$}
    \STATE $symbols = $ \CALL{InitializeSymbols}{$text$} \COMMENT{Pre-populate symbols with individual characters and </w> token}
    \STATE $symbol\:pairs = $ \CALL{InitializeSymbolsPairs}{} 
    \FOR{$i = 0$ \TO $len(symbols) - 1$} \COMMENT{Pre-populate symbol pairs}
        \STATE \CALL{MaybeAddSymbolPair}{$symbol\:pairs$, $merges$, $(symbols[i], symbols[i+1])$}
    \ENDFOR   
    \WHILE {$true$}
        \STATE $(best\:pair, best\:score) = $ \CALL{Pop}{$symbol\:pairs$}
        \IF{$best\:score$ is $null$}
            \BREAK \COMMENT{Symbols can no longer be merged}
        \ENDIF
        \IF{$best\:pair_{left} \neq null$ and $best\:pair_{right} \neq null$}
            \STATE $new\:symbol$ = \CALL{MergeBestPair}{$symbols$, $best\:pair$}
            \STATE $prev = $ \CALL{Left}{$symbols$, $best\:pair_{left}$}
            \STATE $next = $ \CALL{Right}{$symbols$, $best\:pair_{right}$}
            \STATE \CALL{MaybeAddSymbolPair}{$symbol\:pairs$, $merges$, $(prev, best\:pair_{left})$}
            \STATE \CALL{MaybeAddSymbolPair}{$symbol\:pairs$, $merges$, $(best\:pair_{right})$, $next$}
        \ENDIF
    \ENDWHILE
    \RETURN $symbols$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
" %}

Lines _17_ to _21_ initialize both the _Symbols_ and _SymbolPairs_ data structures. At each iteration, the best _SymbolPair_ is popped from the priority queue (line _23_). The check on line _27_ is required as instead of manually removing invalid merges following a merge (if the first 2 elements of a triplets get merged, the pair information for the last 2 elements is no longer valid), we pop pairs from the _SymbolPairs_ and then check their validity. If the pair is still valid, we proceed to merge and check if new pairs should be added (lines _31_ and _32_).

Similarly to the _SymbolsArray_, the Rust implementation for the _Symbols_ Binary Search Tree contains a method mutate itself and merge two symbols:

```rust
pub struct SymbolBTree {
    pub symbols: BTreeSet<Symbol>,
}

impl SymbolBTree {
    pub fn merge_symbols(&mut self, symbol_1: &Symbol, symbol_2: &Symbol) -> Symbol {
        self.symbols.remove(symbol_1);
        self.symbols.remove(symbol_2);
        let new_symbol = Symbol {
            start_byte: symbol_1.start_byte,
            end_byte: symbol_2.end_byte,
        };
        self.symbols.insert(new_symbol);
        new_symbol
    }
}
```

The Tokenizer contains methods to build and maintain a priority queue of _SymbolPairs_ called "agenda":

```rust
impl PriorityQueueBpeTokenizer {
    fn maybe_add_pair(
        &self,
        left_symbol: &Symbol,
        right_symbol: &Symbol,
        input_text: &str,
        agenda: &mut BinaryHeap<SymbolPair>,
    ) {
        let merged_text = &input_text[left_symbol.start_byte..right_symbol.end_byte];
        if let Some(&score) = self.merges_vocab.get(merged_text) {
            agenda.push(SymbolPair {
                left: *left_symbol,
                right: *right_symbol,
                score,
            })
        }
    }
}

impl BpeTokenizer for PriorityQueueBpeTokenizer {
    fn tokenize<'a>(&self, input_text: &'a str) -> Vec<&'a str> {
        let (text, byte_mapping) = self.pre_process_text(input_text, '▁');

        let mut symbols = SymbolBTree::from_text(text.as_str());
        let mut agenda: BinaryHeap<SymbolPair> = BinaryHeap::new();

        for (left_symbol, right_symbol) in symbols.iter().tuple_windows::<(&Symbol, &Symbol)>() {
            self.maybe_add_pair(left_symbol, right_symbol, text.as_str(), &mut agenda);
        }
        
        while let Some(symbol_pair) = agenda.pop() {
            let left_symbol = symbols.get(&symbol_pair.left).cloned();
            let right_symbol = symbols.get(&symbol_pair.right).cloned();

            if let (Some(left_symbol), Some(right_symbol)) = (left_symbol, right_symbol) {
                let new_symbol = symbols.merge_symbols(&left_symbol, &right_symbol);
                if let Some(next) = symbols.symbols.range(new_symbol..).nth(1) {
                    self.maybe_add_pair(&new_symbol, next, text.as_str(), &mut agenda);
                }
                if let Some(prev) = symbols.symbols.range(..new_symbol).next_back() {
                    self.maybe_add_pair(prev, &new_symbol, text.as_str(), &mut agenda);
                }
            }
        }

        let mut output = Vec::new();
        for symbol in symbols {
            output.push(
                &input_text[byte_mapping[&symbol.start_byte]..byte_mapping[&symbol.end_byte]],
            );
        }
        output
    }
}
```

While this is a significant improvement over the naive implementation, accessing the left and right symbols of the best pair for merge (and their predecessor/successor) could be optimized further: a total of 2 `get` operations on the binary search tree and the construction of 2 "throwaway" iterators to find the predecessor and successor likely impact the constant factor of the algorithm. The following section proposes a further optimization keeping the same asymptotic complexity but reducing the number of operations performed at each iteration. 

## d. <a name="pq-ll"></a>Priority Queue + Linked List implementation

The previous implementation solves the problem to identifying the best symbols pair to merge without recalculating the entire list at each iteration, but suffers from inefficiencies to select the symbols to execute the merge. This involves the following operations:
1. Select the left symbol
2. Select the right symbol
3. Create a new symbol, combining left and right
4. Pop the left symbol
5. Pop the right symbol
6. Insert the new symbol
7. Select the left symbol's predecessor, check if it forms a valid pair with the new token
8. Select the right symbol's successor, check if it forms a valid pair with the new token

These operations seem like a natural fit for a linked list: one can easily access predecessors and successors and replace a sequence of arbitrary nodes by a new element. If the `SymbolPair` element contains pointer to its left and right node, one can even skip scanning the linked list to find the nodes, providing $O(1)$ complexity for all th operations above.

Linked lists are however deceptively simple, and are notoriously difficult to implement while both satisfying Rust's borrow checker and offering a high level of performance. An excellent article [[12]](#too-many-inked-lists) highlights the challenges of implementing linked lists in Rust. Instead, Rust's `Vec` growable array data structure has been thoroughly optimized and is recommended over linked lists for better use of CPU cache in the official Rust documentation [[13]](#rust-linked-list).

The following implementation will implement a `LinkedList` behaviour, storing the _Symbol_ nodes in a Rust `Vec`. This is feasible because while new nodes will be inserted following a merge, they will replace 2 local nodes and the linked list will only shrink following its construction. Our _Symbol_  data structure is modified to contain pointers to the previous and next nodes (which are effectively just a position in the "linked list/vec"). This position pointer is a `isize`, where a `-1` value indicates that a given _Symbol_ has no predecessor/successor (first or last node in the list).

```rust
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Symbol {
    pub start_byte: usize,
    pub end_byte: usize,
    pub prev: isize,
    pub next: isize,
    pub size: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct SymbolPair {
    pub left: isize,
    pub right: isize,
    pub score: i64,
    pub pair_size: usize,
}
```

We have also added a `size` field to both the _Symbol_ and the _SymbolPair_. The reason is that following a merge, we will delete the right _Symbol_, and update the left _Symbol_ in-place to represent the combined _Symbol_ (we don't want to grow the linked list). The pointers of the predecessor and successor _Symbols_ are updated to reflect these changes. However, there is no way to find all _SymbolPair_ that contain a given _Symbol_ (the lookup only works in the other direction). This means that following a merge, some _SymbolPair_ (that were pointing to the left element of the previous _SymbolPair_) are no longer valid: the _Symbol_ at this position has changed (its size increased as a result of combining two symbols). We use the `size` information in the _Symbol_ and _SymbolPair_ as a validation step when popping a new _SymbolPair_ from the agenda: if the `size` of the _SymbolPair_ is smaller than the sum of the `size` of its left and right _Symbols_, this _SymbolPair_ is no longer valid and we ignore it (pop the next _SymbolPair_). The size for all _Symbols_ is initialized as `1` when the linked list is constructed from the character list of the text to tokenize.

##### ToDo: add explanatory schema

The data structure holding the _Symbols_ is now a list, implemented using a Rust `Vec`. It implements the method for merging two symbols (mutates itself) from two symbol positions and a size validation (the merge is executed only if the size validation described above succeeds).

```rust
pub struct SymbolList {
    symbols: Vec<Option<Symbol>>,
}

impl SymbolList {
    pub fn merge_symbols(
        &mut self,
        symbol_1_index: usize,
        symbol_2_index: usize,
        size_validation: usize,
    ) -> Option<Symbol> {
        if let (Some(left_symbol), Some(right_symbol)) =
        (self[symbol_1_index], self[symbol_2_index])
        {
            if left_symbol.size + right_symbol.size != size_validation {
                return None;
            }
            if right_symbol.next != -1 {
                if let Some(next_next) = self.symbols.get_mut(right_symbol.next as usize).unwrap() {
                    next_next.prev = symbol_1_index as isize;
                }
            }
            let new_symbol = Symbol {
                start_byte: left_symbol.start_byte,
                end_byte: right_symbol.end_byte,
                prev: left_symbol.prev,
                next: right_symbol.next,
                size: left_symbol.size + right_symbol.size,
            };
            self.symbols[symbol_2_index] = None;
            self.symbols[symbol_1_index] = Some(new_symbol);
            Some(new_symbol)
        } else {
            None
        }
    }
}
```

The tokenizer method looks very similar to the previous implementations. The `maybe_add_pair` method only changes to calculate the size of a _SymbolPair_ from the sum of its _Symbols_ sizes and is skipped below (see the full code at [[7]](#bpe-code)). One notes that following a pop of the minimum of the _SymbolPair_'s priority queue, accessing the left and right token along with their predecessor and successor is now done by directly looking up the fields `left`, `right` of the _SymbolPair_ and `prev`, `next` of the _Symbols_.

```rust
impl BpeTokenizer for PriorityQueueBpeLLTokenizer {
    fn tokenize<'a>(&self, input_text: &'a str) -> Vec<&'a str> {
        let (text, byte_mapping) = self.pre_process_text(input_text, '▁');

        let mut symbols = SymbolList::from_text(text.as_str());
        let mut agenda: BinaryHeap<SymbolPair> = BinaryHeap::new();

        for symbol_index in 1..symbols.len() {
            self.maybe_add_pair(
                symbol_index as isize - 1,
                symbol_index as isize,
                text.as_str(),
                &symbols,
                &mut agenda,
            );
        }

        while let Some(symbol_pair) = agenda.pop() {
            let left_symbol_index = symbol_pair.left;
            let right_symbol_index = symbol_pair.right;
            if left_symbol_index != -1 && right_symbol_index != -1 {
                let new_symbol = symbols.merge_symbols(
                    left_symbol_index as usize,
                    right_symbol_index as usize,
                    symbol_pair.pair_size,
                );
                if let Some(new_symbol) = new_symbol {
                    self.maybe_add_pair(
                        new_symbol.prev,
                        left_symbol_index,
                        text.as_str(),
                        &symbols,
                        &mut agenda,
                    );
                    self.maybe_add_pair(
                        left_symbol_index,
                        new_symbol.next,
                        text.as_str(),
                        &symbols,
                        &mut agenda,
                    );
                }
            }
        }

        let mut output = Vec::new();
        for symbol in symbols {
            if let Some(symbol) = symbol {
                output.push(
                    &input_text[byte_mapping[&symbol.start_byte]..byte_mapping[&symbol.end_byte]],
                );
            }
        }
        output
    }
}
```


## References
- <a name="bpe"></a>[1] [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909), Rico Sennrich, Barry Haddow, Alexandra Birch, 2015 
- <a name="tf-idf"></a>[2] [term frequency–inverse document frequency](https://en.wikipedia.org/wiki/Tf–idf), Wikipedia foundation. 
- <a name="glove"></a>[3] [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf), Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
- <a name="fasttext"></a>[4] [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf), Bojanowski, Grave, Joulin,  Mikolov. 2016.
- <a name="canine"></a>[5] [CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation](https://arxiv.org/abs/2103.06874), Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting, 2021.
- <a name="bpe-lecture"></a>[6] [Natural Language Processing
with Deep Learning, CS224N/Ling284, lecture 12](https://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture12-subwords.pdf), Christopher Manning, Stanford.
- <a name="bpe-code"></a>[7] [Byte Pair Encoding Rust Implementation](https://github.com/guillaume-be/bpe-example), Guillaume Becquin
- <a name="subword-nmt"></a>[8] [Subword Neural Machine Translation](https://github.com/rsennrich/subword-nmt)
- <a name="fastbpe"></a>[9] [fastBPE](https://github.com/glample/fastBPE)
- <a name="sentencepiece-paper"></a>[10] [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226), Taku Kudo, John Richardson
- <a name="sentencepiece-bpe"></a>[11] [SentencePiece BPE model](https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc), Taku Kudo
- <a name="too-many-linked-lists"></a>[12] [Learn Rust With Entirely Too Many Linked Lists](https://rust-unofficial.github.io/too-many-lists/)
- <a name="rust-linked-list"></a>[13] [Rust std LinkedList documentation](https://doc.rust-lang.org/std/collections/struct.LinkedList.html)

