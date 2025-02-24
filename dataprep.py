"""
Data Preparation
=================

This module is responsible for preparing and generating data arrays
required for training the Pyramidal Recursive Neural Network (PyRvNN) model.
It includes functions for streaming input data, converting text into token-level one-hot encodings,
and outputting the necessary data arrays for subsequent processing.
"""

from globalvars import *
from fstreamer import stream

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
import re


tokenizer = Tokenizer(BPE())
tokenizer = tokenizer.from_file(TOKENIZER_FILENAME)


def _string_to_tokenIDs(string):
    bpe_output = tokenizer.encode([string], is_pretokenized=True)
    tokenIDs = np.array( bpe_output.ids )
    return tokenIDs


def _tokenIDs_to_onehots(tokenIDs):
    onehots = np.zeros((len(tokenIDs), len(VOCAB)))
    try:
        onehots[np.arange(len(tokenIDs)), tokenIDs] = 1
    except:
        print("tokenIDs:", tokenIDs)
    return onehots


def text_to_words_onehots(text):
    """
    Converts input text into an array of one-hot encoded tokens for each word.

    The function tokenizes the input text into words, applies Byte Pair Encoding (BPE),
    and converts the tokenized words into one-hot encoded vectors. It ensures each word 
    is wrapped with special tokens indicating the start and end of a word.

    Parameters
    ----------
    text : str
        The input text to be tokenized and converted into one-hot encodings.

    Returns
    -------
    list of numpy.ndarray
        A list of one-hot encoded representations, where each element corresponds
        to a word in the input text.
    """

    text = re.sub(r'([^ ])-([^ ])', r'\1 <-> \2', text)
    words = list(filter(None, text.split(" ")))
    words_tokenIDs = [ [] for i in range(len(words)) ]
    for i in range(len(words)):
        words_tokenIDs[i] = np.concatenate(( [TOKEN_TO_ID["<word.beg>"]], _string_to_tokenIDs(words[i]), [TOKEN_TO_ID["<word.end>"]] ))
    words_onehots = [ _tokenIDs_to_onehots(word_tokenIDs) for word_tokenIDs in words_tokenIDs ]
    words_onehots = list(filter(lambda x: x != [], words_onehots))
    return words_onehots


def generate_arrays_from_data(init_index_pos):
    """
    Streams text data and converts it into word-level one-hot encoded arrays.

    The function reads lines from a dataset file, tokenizes the text, and
    outputs word-level one-hot encoded representations. It ensures lines
    exceeding a certain length are skipped.

    Parameters
    ----------
    init_index_pos : int
        The initial index position in the dataset file to start streaming from.

    Yields
    ------
    tuple
        A tuple containing:
        - list of numpy.ndarray: One-hot encoded word representations.
        - int: The updated index position in the dataset file.
    """

    last_index_pos = init_index_pos
    
    line_num = 0
    while 1:
        line_num += 1
        line, last_index_pos = stream(DATASET, INDEX_FILE, last_index_pos)
        while(len(line) > 2000):
            print("  ---- skipped a line of text ----")
            line, last_index_pos = stream(DATASET, INDEX_FILE, last_index_pos)
        if last_index_pos == "end":
            last_index_pos = 0
            line_num = 0
        else:
            #print("================", line)
            words_onehots = text_to_words_onehots(line)
            yield words_onehots, last_index_pos
