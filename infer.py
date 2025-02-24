from globalvars import *
from dataprep import text_to_words_onehots
import datamanip

import numpy as np


def onehots_to_chars(onehots):
    """
    This function decodes two sets of one-hot encoded vectors into corresponding character arrays.

    Parameters
    ----------
    onehots : list of numpy.ndarray
        A list containing two numpy arrays of one-hot encoded vectors.

    Returns
    -------
    tuple
        A tuple containing:
        - numpy.ndarray: Decoded characters from the first one-hot encoded array.
        - numpy.ndarray: Decoded characters from the second one-hot encoded array.
    """

    max_indices_0 = onehots[0].argmax(axis=1)
    max_indices_1 = onehots[1].argmax(axis=1)
    chars_0 = np.array([ VOCAB[i] for i in max_indices_0 ])
    chars_1 = np.array([ VOCAB[i] for i in max_indices_1 ])
    return chars_0, chars_1


def decode_phraselvl_to_string(model, embs, word_lengths, depth):
    """
    Decodes phrase-level embeddings into a string representation.

    Parameters
    ----------
    model : object
        The model (PyRvNN) used for decoding.
    embs : numpy.ndarray
        The embeddings at the phrase level.
    word_lengths : list of int
        The lengths of words in the original text.
    depth : int
        The depth at which decoding should occur.

    Returns
    -------
    numpy.ndarray
        The decoded character sequence.
    """

    if depth > 0:
        embs = _decode_phraselvl_to_wordlvl(model, embs, depth)
    onehots = _decode_wordlvl_to_logits(model, embs, word_lengths)

    chars = onehots_to_chars(onehots)[0]
    return chars


def _decode_phraselvl_to_wordlvl(model, embs, depth):
    depth_i = 0
    while depth_i < depth:
        depth_i += 1
        
        outputs = model.decode_enc(embs)
        embs = np.concatenate( (outputs["dense_left"].numpy(), outputs["dense_right"].numpy()), axis=1 )
        embs = _combine_embs_phraselvl(embs)
    
    return embs


def _decode_wordlvl_to_logits(model, embs, word_lengths, depth=0):
    bottom_embs = [[] for i in range(len(word_lengths))] #np.array([])
    
    while True:
        depth += 1

        embs_to_decode = np.array([])
        prev_end_index = 0
        for i, word_length in enumerate(word_lengths):
            if len(bottom_embs[i]):
                continue
            word_subembs = embs[prev_end_index:prev_end_index+depth]
            prev_end_index = prev_end_index + depth
            if depth < word_length-1:  # if didnt reach the bottom embs (pre-hotones) of the word
                if len(embs_to_decode) == 0:
                    embs_to_decode = word_subembs
                else:
                    embs_to_decode = np.append(embs_to_decode, word_subembs, axis=0)
            else:  # save bottom embs (pre-hotones) of the word
                bottom_embs[i] = word_subembs
   
        if len(embs_to_decode) == 0:
            break
        else:
            outputs = model.decode_enc(embs_to_decode)
            embs = np.concatenate( (outputs["dense_left"].numpy(), outputs["dense_right"].numpy()), axis=1 )
            num_words = len(word_lengths) - sum([ 1 for x in bottom_embs if len(x) ])
            embs = _combine_embs_subwordlvl(embs, depth, num_words)
    
    bottom_embs_flat = np.array([ item for sublist in bottom_embs for item in sublist ])
    outputs = model.decode_enc(bottom_embs_flat)
    onehots = [ outputs["onehot_left"].numpy(), outputs["onehot_right"].numpy() ]
    return onehots  # wannabe hot pairs, but not really :D


def _combine_embs_phraselvl(embs):
    combined_embs = np.array([])
    prev_end_index = 0

    edge_left_node = embs[0,:EMBEDDING_SIZE]
    edge_right_node = embs[-1,EMBEDDING_SIZE:]

    node_rep_pairs = np.zeros(( len(embs)-1, 2, EMBEDDING_SIZE ))  # every node now has two embs that represent it
    node_embs_combined = np.zeros(( len(embs)-1, EMBEDDING_SIZE ))  # testing
    for i in range(len(node_rep_pairs)):
        #node_rep_pairs[i,0] = word_subembs[i,EMBEDDING_SIZE:]
        #node_rep_pairs[i,1] = word_subembs[i+1,:EMBEDDING_SIZE]
        node_embs_combined[i] = embs[i,EMBEDDING_SIZE:]  # testing
            
    #node_embs_combined = node_rep_pairs.mean(axis=1)
    node_embs_combined = np.insert( node_embs_combined, 0, edge_left_node, axis=0 )
    node_embs_combined = np.insert( node_embs_combined, len(node_embs_combined), edge_right_node, axis=0 )
        
    if len(combined_embs) == 0:
        combined_embs = node_embs_combined
    else:
        combined_embs = np.append(combined_embs, node_embs_combined, axis=0)
            
    return combined_embs


def _combine_embs_subwordlvl(embs, depth, num_words):
    word_length = depth
    combined_embs = np.array([])
    prev_end_index = 0
    for i in range(num_words):
        word_subembs = embs[prev_end_index:prev_end_index+word_length]
        prev_end_index = prev_end_index + word_length
        
        edge_left_node = word_subembs[0,:EMBEDDING_SIZE]
        edge_right_node = word_subembs[-1,EMBEDDING_SIZE:]

        node_rep_pairs = np.zeros(( len(word_subembs)-1, 2, EMBEDDING_SIZE ))  # every node now has two embs that represent it
        node_embs_combined = np.zeros(( len(word_subembs)-1, EMBEDDING_SIZE ))  # testing
        for i in range(len(node_rep_pairs)):
            #node_rep_pairs[i,0] = word_subembs[i,EMBEDDING_SIZE:]
            #node_rep_pairs[i,1] = word_subembs[i+1,:EMBEDDING_SIZE]
            node_embs_combined[i] = word_subembs[i,EMBEDDING_SIZE:]  # testing
            
        #node_embs_combined = node_rep_pairs.mean(axis=1)
        node_embs_combined = np.insert( node_embs_combined, 0, edge_left_node, axis=0 )
        node_embs_combined = np.insert( node_embs_combined, len(node_embs_combined), edge_right_node, axis=0 )
        
        if len(combined_embs) == 0:
            combined_embs = node_embs_combined
        else:
            combined_embs = np.append(combined_embs, node_embs_combined, axis=0)
            
    return combined_embs


def text_emb_lvl(model, text, lvl, multilvl=False):   # phrase / sentence / line of text
    """
    This function processes text data through a model to obtain embeddings 
    at different levels (token, subword, word, phrase). It supports multi-level 
    encoding.

    Parameters
    ----------
    model : object
        The model (PyRvNN) used for encoding.
    text : str
        The input text to be encoded.
    lvl : int
        The hierarchical level at which to return embeddings.
    multilvl : bool, optional
        Whether to return embeddings at multiple levels (default is False).

    Returns
    -------
    numpy.ndarray or list
        The encoded embeddings at the specified level, or a list of embeddings 
        if multi-level encoding is enabled.
    """

    input_data = text_to_words_onehots(text)

    word_lengths = [ word.shape[0] for word in input_data ]
    word_lvl_lengths = word_lengths.copy()
    word_embs = np.zeros(( len(input_data), EMBEDDING_SIZE )) # storage for word embs while all words dont have an emb

    subword_depth = 0
    phrase_depth = 0
    below_wordlvl = True

    multilvl_embs = []
    
    while True:
        
        if below_wordlvl:  # subword level
            if subword_depth == 0:  # tokens (input data)
                enc_pairs, reg_pairs = datamanip.token_pairs_prep(input_data)
            else:  # subwords (above tokens, below words)
                enc_pairs, reg_pairs, word_lvl_lengths, word_embs, reached_words = datamanip.subword_pairs_prep(embs, word_lvl_lengths, word_embs, subword_depth)
                if reached_words:   # reached words (phrase level 0)
                    below_wordlvl = False
                    embs = word_embs
                    if multilvl:
                        multilvl_embs.append(word_embs)
                    if lvl == 0:
                        if multilvl:
                            return multilvl_embs
                        return word_embs
                    continue
            subword_depth += 1
        
        else:  # phrase levels (word level and above)
            if len(embs) < 2:
                break
            enc_pairs, reg_pairs = datamanip.phraselvl_pairs_prep(embs)
            phrase_depth += 1
        
        depth = (subword_depth) if below_wordlvl else (phrase_depth)
        y = np.concatenate( (enc_pairs, reg_pairs), axis=1 )
        embs = model.encode(enc_pairs)

        if multilvl:
            if below_wordlvl and subword_depth > 1:
                multilvl_embs[0].append(embs)
            else:
                multilvl_embs.append([embs])

        if phrase_depth == lvl:
            if multilvl:
                return multilvl_embs
            return embs
