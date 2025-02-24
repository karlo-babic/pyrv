"""
Data Manipulation
=================

This module facilitates the preparation of input and output data pairs at various levels
of the pyramidal hierarchy in the PyRv method.
It includes functions for organizing and pairing data at the token, subword, and phrase levels,
preparing the inputs and outputs required for both the autoencoding and autoregressive tasks.
"""

from globalvars import *
import numpy as np


def _make_pairs(units):
    unit_pairs = np.concatenate( (units[:-1], units[1:]), axis=1 )
    return unit_pairs


def _make_neighbors(units):
    unit_size = units.shape[1]
    units_with_paddings = np.concatenate( [[np.zeros(unit_size)], units, [np.zeros(unit_size)]], axis=0 )
    unit_neighbors = np.concatenate( (units_with_paddings[:-3], units_with_paddings[3:]), axis=1 )
    return unit_neighbors


def token_pairs_prep(input_data):
    """
    Prepares input and output token pairs for autoencoding and autoregression.

    Parameters
    ----------
    input_data : list of numpy.ndarray
        A list of one-hot encoded token representations.

    Returns
    -------
    tuple
        A tuple containing:
        - numpy.ndarray: Token pairs for autoencoding head, with placeholders for dense embeddings.
        - numpy.ndarray: Token neighbor pairs for autoregression head, with placeholders for dense embeddings.
    """

    # creating the input (and enc output) "oneHot+emptyDense" vector
    enc_pairs = [ _make_pairs(token_onehots) for token_onehots in input_data ]
    enc_pairs = np.concatenate(enc_pairs, axis=0)
    enc_pairs = np.concatenate( (enc_pairs, np.zeros((enc_pairs.shape[0], EMBEDDING_SIZE*2))), axis=1 )
    # creating the reg output "oneHot+emptyDense" vector
    reg_pairs = [ _make_neighbors(token_onehots) for token_onehots in input_data ]
    reg_pairs = np.concatenate(reg_pairs, axis=0)
    reg_pairs = np.concatenate( (reg_pairs, np.zeros((reg_pairs.shape[0], EMBEDDING_SIZE*2))), axis=1 )

    return enc_pairs, reg_pairs


def subword_pairs_prep(embs, word_lvl_lengths, word_embs, subword_depth):
    """
    Prepares input and output pairs of subword embeddings for autoencoding and autoregression,
    and augments them with special one-hot markers indicating subword-level processing.

    Parameters
    ----------
    embs : numpy.ndarray
        The input embeddings to be processed.
    word_lvl_lengths : list of int
        The lengths of word-level representations (number of subword embs in each word).
    word_embs : numpy.ndarray
        The embeddings at the word level.
    subword_depth : int
        The depth at which subword representations are processed.

    Returns
    -------
    tuple
        A tuple containing:
        - numpy.ndarray or None: Subword pairs (for autoencoding) if available, else None.
        - numpy.ndarray or None: Subword neighbor pairs (for autoregression) if available, else None.
        - list of int or None: Updated word-level lengths or None if processing is complete.
        - numpy.ndarray: Updated word embeddings.
        - bool: A flag indicating whether word-level is reached.
    """

    embs_tokenized = []
    prev_end_index = 0

    for i, tok_len in enumerate(word_lvl_lengths):
        if word_lvl_lengths[i] == 0:
            continue
        subword_embs = embs[prev_end_index:prev_end_index+(tok_len-subword_depth)]
        if len(subword_embs) == 1:  # if merged up all the way to the one representation of the token
            word_embs[i] = subword_embs[0]
            embs = np.delete(embs, prev_end_index, 0)
            word_lvl_lengths[i] = 0
        else:
            embs_tokenized.append( subword_embs )
            prev_end_index = prev_end_index + (tok_len-subword_depth)

    if len(embs_tokenized) > 0:  # if there are some representations of subwords left, make pairs and neighbors of subword representations
        # creating the input (and enc output) "emptyOneHot+Dense" vector
        #enc_pairs = np.array([ _make_pairs(tok_embs) for tok_embs in embs_tokenized ])
        enc_pairs = list( map(_make_pairs, embs_tokenized) )
        enc_pairs = np.concatenate(enc_pairs, axis=0)
        
        empty_onehot = np.zeros((enc_pairs.shape[0], len(VOCAB)*2))
        empty_onehot[:, TOKEN_TO_ID["<subwordlvl>"]] = 1
        empty_onehot[:, len(VOCAB) + TOKEN_TO_ID["<subwordlvl>"]] = 1
        enc_pairs = np.concatenate( (empty_onehot, enc_pairs), axis=1 )

        # creating the reg output "emptyOneHot+Dense" vector
        #reg_pairs = np.array([ _make_neighbors(tok_embs) for tok_embs in embs_tokenized ])
        reg_pairs = list( map(_make_neighbors, embs_tokenized) )

        reg_pairs = np.concatenate(reg_pairs, axis=0)
        reg_pairs = np.concatenate( (empty_onehot, reg_pairs), axis=1 )

        return enc_pairs, reg_pairs, word_lvl_lengths, word_embs, False
    
    else:  # reached word level
        return None,      None,      None,          word_embs, True

    
def phraselvl_pairs_prep(embs):
    """
    This function takes embeddings at the phrase level, constructs their 
    pairs and neighbors, and augments them with special one-hot markers 
    indicating phrase-level processing.

    Parameters
    ----------
    embs : numpy.ndarray
        The input embeddings at the phrase level.

    Returns
    -------
    tuple
        A tuple containing:
        - numpy.ndarray: Phrase-level pairs (for autoencoding).
        - numpy.ndarray: Phrase neighbor pairs (for regressionencoding).
    """

    # creating the input (and enc output) "emptyOneHot+Dense" vector
    enc_pairs = _make_pairs(embs)
    empty_onehot = np.zeros((enc_pairs.shape[0], len(VOCAB)*2))
    empty_onehot[:, TOKEN_TO_ID["<phraselvl>"]] = 1
    empty_onehot[:, len(VOCAB) + TOKEN_TO_ID["<phraselvl>"]] = 1
    enc_pairs = np.concatenate( (empty_onehot, enc_pairs), axis=1 )
    
    # creating the reg output "emptyOneHot+Dense" vector
    reg_pairs = _make_neighbors(embs)
    reg_pairs = np.concatenate( (empty_onehot, reg_pairs), axis=1 )
    
    return enc_pairs, reg_pairs
