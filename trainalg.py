"""
Training Algorithm
==================

This module encapsulates the core training algorithms,
including pyramidal recursion and optimization processes.
"""

from globalvars import *
import dataprep
import datamanip
import infer
import vis

import tensorflow as tf
import numpy as np
from time import time


class Monitor():
    """
    A utility class for tracking and managing training progress.

    This class maintains logs of training statistics, such as losses and 
    model performance, to facilitate monitoring and debugging. It provides 
    mechanisms to store, retrieve, and update training-related data without 
    affecting the core training process.

    Attributes:
    -----------
    Various internal attributes to track losses, metrics, and other 
    relevant training details.
    """

    def __init__(self) -> None:
        self.model = None
        self.print_delay_seconds = 600
        self.steps = []
        self.prev_time = time()
        self.do_monitor = True
        self.losses = []
        self.evals = []
        self.vis_timer = 0
        self.do_visualize = False
        
    def tick(self, step):
        if time() - self.prev_time >= self.print_delay_seconds:
            self.do_monitor = True
            self.prev_time = time()
            self.steps.append(step)
            
    def record_losses(self, total_losses):
        if not self.do_monitor: return
        self.losses.append(total_losses)
        
    def eval(self, lvl, args):
        if not self.do_monitor: return
        if lvl == 'tokenlvl':
            enc_pairs, reg_pairs, depth = args
            self._evaluate_tokenlvl(enc_pairs, reg_pairs)
        elif lvl == 'subwordlvl':
            depth, losses = args
            self._evaluate_subwordlvl(depth, losses)
        elif lvl == 'wordlvl':  # phrase lvl 0
            embs, word_lengths, losses = args
            self._eval_phraselvl(embs, word_lengths, 0, losses)
        elif lvl == 'phraselvl':
            embs, word_lengths, depth, losses = args
            self._eval_phraselvl(embs, word_lengths, depth, losses)
            
    def display(self, step, loss):
        print('1.0 mini','-'*32, step, '    loss:', loss.numpy(), '      ', '\r', end='')
        if not self.do_monitor: return
        self.do_monitor = False
        print('\n')
        
        for eval in self.evals:
            print(eval)
        
        print('\n\tTOTAL LOSSES')
        print(f"loss:\t{self.losses[-1]['total']:.4f}")
        print("\tenc:\treg:")
        print(f"onehot:\t{self.losses[-1]['onehot_enc']:.4f}\t{self.losses[-1]['onehot_reg']:.4f}")
        print(f"dense:\t{self.losses[-1]['dense_enc']:.4f}\t{self.losses[-1]['dense_reg']:.4f}")
        print(f"negative reg:\t{self.losses[-1]['negative_reg']:.4f}")
        print('\n')

        self.vis_timer += 1
        if self.do_visualize and (self.vis_timer == 10 or step == 0):
            self.vis_timer = 0
            print("visualizing...")
            print("    embedding")
            embs = infer.text_emb_lvl(self.model, vis.text, 2, multilvl=True)
            print("    reducing dimensions")
            vis.plot_embeddings(embs, vis.text, which_lvls=[1,1,0,0], filename="vis/vis_embs_sw_"+str(step)+".png", annotate=True)
            vis.plot_embeddings(embs, vis.text, which_lvls=[0,1,1,1], filename="vis/vis_embs_wp_"+str(step)+".png", annotate=True)
            print('\n\n')
    
    def _evaluate_tokenlvl(self, enc_pairs, reg_pairs):
        # enc token-level accuracy
        onehots = [ enc_pairs[:,:len(VOCAB)], enc_pairs[:,len(VOCAB):len(VOCAB)*2] ]
        enc_input_chars = infer.onehots_to_chars(onehots)
        outputs = self.model(enc_pairs)
        onehots = [outputs["enc"]["onehot_left"].numpy(), outputs["enc"]["onehot_right"].numpy()]
        enc_output_chars = infer.onehots_to_chars(onehots)
        enc_char_pairs_accuracy = ( (enc_input_chars[0]==enc_output_chars[0]).sum() + (enc_input_chars[1]==enc_output_chars[1]).sum() ) / ( len(enc_input_chars[0])*2 )
        # reg char-level accuracy
        onehots = [ reg_pairs[:,:len(VOCAB)], reg_pairs[:,len(VOCAB):len(VOCAB)*2] ]
        reg_input_chars = infer.onehots_to_chars(onehots)
        outputs = self.model(enc_pairs)
        onehots = [outputs["reg"]["onehot_left"].numpy(), outputs["reg"]["onehot_right"].numpy()]
        reg_output_chars = infer.onehots_to_chars(onehots)
        reg_char_pairs_accuracy = ( (reg_input_chars[0]==reg_output_chars[0]).sum() + (reg_input_chars[1]==reg_output_chars[1]).sum() ) / ( len(reg_input_chars[0])*2 )
        self.evals = [ f"\tTOKEN LVL ACCURACY\nenc:{enc_char_pairs_accuracy: .2%}\nreg:{reg_char_pairs_accuracy: .2%}" ]

    def _evaluate_subwordlvl(self, depth, losses):
        print_string = ""
        if depth == 1:
            print_string = "\n\tSUBWORD\t\tenc:\t\treg:\t\tloss:\n"
        print_string += f"\t\tLVL {depth}\t{losses['loss_dense_enc']:.6f}\t{losses['loss_dense_reg']:.6f}\t{losses['loss']:.6f}"
        self.evals.append(print_string)

    def _eval_phraselvl(self, embs, word_lengths, depth, losses):
        chars = infer.decode_phraselvl_to_string(self.model, embs, word_lengths, depth)
        chars_out = ''.join(chars).replace("<word.beg>", " ").replace("<word.end>", " ").replace("<subwordlvl>", "_").replace("<phraselvl>", "_")

        print_string = ""
        if depth == 0:
            print_string = f"\n\tWORD\n"
        elif depth == 1:
            print_string = f"\n\tPHRASE\n"
        print_string += f"\t\tLVL {depth}\t{losses['loss_dense_enc']:.6f}\t{losses['loss_dense_reg']:.6f}\t{losses['loss']:.6f}\n{chars_out}"
        self.evals.append(print_string)

monitor = Monitor()



loss_object_onehot = tf.keras.losses.BinaryCrossentropy()
loss_object_dense = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
def loss(model, x, y, training, depth, below_wordlvl, total_loss):
    """
    Compute the loss for the given input and target data.

    Parameters:
    -----------
    model : tf.keras.Model
        The neural network model.
    x : np.ndarray
        Input data.
    y : tf.Tensor
        Target data converted to a tensor.
    training : bool
        Indicates whether the model is in training mode.
    depth : int
        The current depth level in the hierarchical structure.
    below_wordlvl : bool
        Flag indicating if the training is at the subword level.
    total_loss : float
        Cumulative total loss from previous steps.

    Returns:
    --------
    dict
        A dictionary containing different components of the loss.
    np.ndarray
        The computed embeddings.
    """

    outputs, embs = model(x, training=training, return_embs=True)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    
    y_onehot_enc_left  = y[:,:len(VOCAB)]
    y_onehot_enc_right = y[:,len(VOCAB):len(VOCAB)*2]
    y_dense_enc_left  = y[:,len(VOCAB)*2:int(model.output_dim/2)-EMBEDDING_SIZE]
    y_dense_enc_right = y[:,int(model.output_dim/2)-EMBEDDING_SIZE:int(model.output_dim/2)]
    
    y_onehot_reg_left  = y[:,int(model.output_dim/2):int(model.output_dim/2)+len(VOCAB)]
    y_onehot_reg_right = y[:,int(model.output_dim/2)+len(VOCAB):int(model.output_dim/2)+len(VOCAB)*2]
    y_dense_reg_left  = y[:,-EMBEDDING_SIZE*2:-EMBEDDING_SIZE]
    y_dense_reg_right = y[:,-EMBEDDING_SIZE:]
    
    if below_wordlvl:
        reg_norm_depth = ( (depth+0.001) / (depth+20) ) * 0.1
        enc_norm_depth = 1 / (depth+1)
    else:
        reg_norm_depth = 1#( (depth+0.001) / (depth+20) ) #* 0.1
        enc_norm_depth = 0.05
    
    #do_enc = step % 2
    #do_reg = abs(do_enc - 1)
    
    loss_onehot_enc_left  = loss_object_onehot(y_onehot_enc_left,  outputs["enc"]["onehot_left"])  * enc_norm_depth
    loss_onehot_enc_right = loss_object_onehot(y_onehot_enc_right, outputs["enc"]["onehot_right"]) * enc_norm_depth
    loss_onehot_enc = (loss_onehot_enc_left + loss_onehot_enc_right) / 2  #* do_enc

    loss_onehot_reg_left  = loss_object_onehot(y_onehot_reg_left,  outputs["reg"]["onehot_left"])  * reg_norm_depth
    loss_onehot_reg_right = loss_object_onehot(y_onehot_reg_right, outputs["reg"]["onehot_right"]) * reg_norm_depth
    loss_onehot_reg = (loss_onehot_reg_left + loss_onehot_reg_right) / 2  #* do_reg
    
    loss_dense_enc_left  = loss_object_dense(y_dense_enc_left,  outputs["enc"]["dense_left"])  * enc_norm_depth
    loss_dense_enc_right = loss_object_dense(y_dense_enc_right, outputs["enc"]["dense_right"]) * enc_norm_depth
    loss_dense_enc = (loss_dense_enc_left + loss_dense_enc_right) / 2  #* do_enc
    
    loss_dense_reg_left  = loss_object_dense(y_dense_reg_left,  outputs["reg"]["dense_left"])  * reg_norm_depth
    loss_dense_reg_right = loss_object_dense(y_dense_reg_right, outputs["reg"]["dense_right"]) * reg_norm_depth
    loss_dense_reg = (loss_dense_reg_left + loss_dense_reg_right) / 2  #* do_reg

    #loss_negative_onehot_reg_left  = loss_object_dense(y_onehot_reg_left,  outputs["reg"]["onehot_right"]) * reg_norm_depth * 0.01
    #loss_negative_onehot_reg_right = loss_object_dense(y_onehot_reg_right, outputs["reg"]["onehot_left"])  * reg_norm_depth * 0.01
    #loss_negative_dense_reg_left  = loss_object_dense(y_dense_reg_left,  outputs["reg"]["dense_right"]) * reg_norm_depth * 0.01
    #loss_negative_dense_reg_right = loss_object_dense(y_dense_reg_right, outputs["reg"]["dense_left"])  * reg_norm_depth * 0.01
    #loss_negative_reg = - (loss_negative_onehot_reg_left + loss_negative_onehot_reg_right + loss_negative_dense_reg_left + loss_negative_dense_reg_right) / 4
    loss_negative_reg = 0#1/(1+np.e**(-loss_negative_reg))  # sigmoid
    
    loss = (loss_onehot_enc + loss_onehot_reg + loss_dense_enc + loss_dense_reg + loss_negative_reg)  #/ 4

    if below_wordlvl and depth == 1:
        magnitude = 1
    else:
        magnitude_enc_left = tf.norm(y_dense_enc_left, axis=1)
        magnitude_enc_right = tf.norm(y_dense_enc_right, axis=1)
        magnitude_reg_left = tf.norm(y_dense_reg_left, axis=1)
        magnitude_reg_right = tf.norm(y_dense_reg_right, axis=1)
        magnitude = (magnitude_enc_left + magnitude_enc_right + magnitude_reg_left + magnitude_reg_right) / 4
        magnitude = np.mean(magnitude)
    
    loss /= magnitude
    loss /= x.shape[0]  # devided by the number of inputs or outpus loss is calculated on
    total_loss_normalize = 1 / ( (total_loss*10**1)**6 + 1 )
    loss *= total_loss_normalize
    
    return {"loss": loss,
            "loss_onehot_enc": loss_onehot_enc,
            "loss_onehot_reg": loss_onehot_reg,
            "loss_dense_enc": loss_dense_enc,
            "loss_dense_reg": loss_dense_reg,
            "negative_reg": loss_negative_reg}, embs


def pyramidal_recursion(model, input_data, max_subword_depth, max_phrase_depth):
    """
    Perform hierarchical training using a pyramidal recursion approach.

    Parameters:
    -----------
    model : tf.keras.Model
        The neural network model.
    input_data : list
        List of input sequences to process.
    max_subword_depth : int
        Maximum depth for subword-level processing.
    max_phrase_depth : int
        Maximum depth for phrase-level processing.

    Returns:
    --------
    float
        The total loss computed during the recursion.
    """

    total_losses = {
        'total': 0,
        'onehot_enc': 0,
        'onehot_reg': 0,
        'dense_enc': 0,
        'dense_reg': 0,
        'negative_reg': 0
    }
    
    word_lengths = [ word.shape[0] for word in input_data ]
    word_lvl_lengths = word_lengths.copy()
    word_embs = np.zeros(( len(input_data), EMBEDDING_SIZE )) # storage for word embs while all words dont have a rep

    subword_depth = 0
    phrase_depth = 0
    below_wordlvl = True
    
    while (below_wordlvl and subword_depth < max_subword_depth)\
        or (not below_wordlvl and phrase_depth < max_phrase_depth):
        
        if below_wordlvl:  # subword level
            if subword_depth == 0:  # tokens (input data)
                enc_pairs, reg_pairs = datamanip.token_pairs_prep(input_data)
            else:  # subwords (above tokens, below words)
                enc_pairs, reg_pairs, word_lvl_lengths, word_embs, reached_words = datamanip.subword_pairs_prep(embs, word_lvl_lengths, word_embs, subword_depth)
                if reached_words:   # reached words (phrase level 0)
                    below_wordlvl = False
                    embs = word_embs
                    monitor.eval('wordlvl', (embs, word_lengths, losses))
                    continue
            subword_depth += 1
        
        else:  # phrase levels (word level and above)
            if len(embs) < 2:
                break
            enc_pairs, reg_pairs = datamanip.phraselvl_pairs_prep(embs)
            phrase_depth += 1
        
        depth = (subword_depth) if below_wordlvl else (phrase_depth)
        y = np.concatenate( (enc_pairs, reg_pairs), axis=1 )
        losses, embs = loss(model, enc_pairs, y, True, depth, below_wordlvl, total_losses['total'])
        update_total_losses(losses, total_losses)

        if below_wordlvl:
            if depth == 1:  # decode to tokens (input data)
                monitor.eval('tokenlvl', (enc_pairs, reg_pairs, depth))
            else:
                monitor.eval('subwordlvl', (depth-1, losses))
        else:
            monitor.eval('phraselvl', (embs, word_lengths, depth, losses))
    
    total_losses['total'] /= (subword_depth + phrase_depth)
    monitor.record_losses(total_losses)
    
    return total_losses['total']


def update_total_losses(losses, total_losses):
    """
    Update the cumulative loss dictionary with new loss values.

    Parameters:
    -----------
    losses : dict
        A dictionary containing the latest computed losses.
    total_losses : dict
        A dictionary tracking the accumulated losses.
    """

    total_losses['total'] += losses["loss"]
    total_losses['onehot_enc'] += losses["loss_onehot_enc"]
    total_losses['onehot_reg'] += losses["loss_onehot_reg"]
    total_losses['dense_enc'] += losses["loss_dense_enc"]
    total_losses['dense_reg'] += losses["loss_dense_reg"]
    total_losses['negative_reg'] += losses["negative_reg"]


def train_init(model, optimizer, learning_rate):
    """
    Initialize the training process, loading previous state if available.

    Parameters:
    -----------
    model : tf.keras.Model
        The neural network model.
    optimizer : tf.keras.optimizers.Optimizer or None
        The optimizer for training. If None, an Adam optimizer is created.
    learning_rate : float
        The learning rate for training.

    Returns:
    --------
    tuple
        A tuple containing:
        - optimizer (tf.keras.optimizers.Optimizer): The optimizer to use.
        - init_step (int): The starting step for training.
        - data_gen (generator): The data generator for training batches.
        - do_train (bool): Whether training should proceed (False if loading state failed).
    """

    do_train = True
    if not optimizer:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False, name='Adam')
    try:
        with open(SAVED_STATE) as file:
            prev_state = file.readline().split()
            init_step = int(prev_state[0])
            last_index_pos = int(prev_state[1])
    except Exception as e:
        print(f"An error occurred while reading the file {SAVED_STATE}: {e}")
        do_train = False
    data_gen = dataprep.generate_arrays_from_data( last_index_pos )
    monitor.model = model
    return optimizer, init_step, data_gen, do_train

def pyrv_train(model, max_subword_depth, max_phrase_depth, num_steps, learning_rate, optimizer=None):
    """
    Train the model using pyramidal recursion for a specified number of steps.

    Parameters:
    -----------
    model : tf.keras.Model
        The neural network model.
    max_subword_depth : int
        Maximum depth for subword-level processing.
    max_phrase_depth : int
        Maximum depth for phrase-level processing.
    num_steps : int
        Number of training steps.
    learning_rate : float
        Learning rate for training.
    optimizer : tf.keras.optimizers.Optimizer, optional
        The optimizer for training. If None, an Adam optimizer is created.

    Writes:
    -------
    A file "saved_state.txt" to store the training state.
    """
    
    optimizer, init_step, data_gen, do_train = train_init(model, optimizer, learning_rate)
    step = init_step

    ## Iterating through the dataset
    while do_train and step < init_step + num_steps:
        monitor.tick(step)
        input_data, last_index_pos = next(data_gen)
        
        with tf.GradientTape(persistent=False) as tape:
            loss_total_pyramid = pyramidal_recursion(model, input_data, max_subword_depth, max_phrase_depth)

        monitor.display(step, loss_total_pyramid)

        if loss_total_pyramid <= 1.2:
            grads = tape.gradient(loss_total_pyramid, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        step += 1

    with open("saved_state.txt", "w") as f:
        f.write( str(step) + " " + str(last_index_pos) )
