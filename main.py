"""
Main Script
===========

Serves as the entry point for orchestrating the training of the PyRv model (`python main.py`).
It defines a progressive training schedule that incrementally exposes the model
to deeper pyramidal hierarchies in a staged manner.
This schedule ensures efficient and stable learning by carefully managing the depth
of subword and phrase pyramid levels, as well as adjusting the learning rate across training stages.
"""

from globalvars import *
from modelarch import PyRvNN
from trainalg import pyrv_train
import os


name = 'a'
checkpoint = 0

model = PyRvNN()
checkpoints_folder = "checkpoints"
if not os.path.exists(checkpoints_folder):
    os.makedirs(checkpoints_folder)
if checkpoint > 0:
    model.load_weights('checkpoints/' + name + str(checkpoint))



pyrv_train(model, max_subword_depth=1, max_phrase_depth=0, num_steps=10000, learning_rate=0.001)

model.save_weights('./checkpoints/' + name + str(1))
print("==== subword lvl 0 DONE, saved checkpoint")

for i in range(2, 10):
    pyrv_train(model, max_subword_depth=i, max_phrase_depth=0, num_steps=2000, learning_rate=0.001)
    print(f"==== subword lvl {i} DONE")

model.save_weights('./checkpoints/' + name + str(2))
print("==== subword lvls DONE, saved checkpoint")

pyrv_train(model, max_subword_depth=10, max_phrase_depth=1, num_steps=10000, learning_rate=0.001)

model.save_weights('./checkpoints/' + name + str(3))
print("==== phrase lvl 1 DONE, saved checkpoint")

for i in range(1, 3):
    pyrv_train(model, max_subword_depth=10, max_phrase_depth=2, num_steps=100000, learning_rate=0.0005)
    model.save_weights('./checkpoints/' + name + str(i+3))
    print(f"==== phrase lvl 2 DONE (iter {i}), saved checkpoint")