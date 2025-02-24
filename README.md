# PyRvNN: Pyramidal Recursive Neural Network for Text Representation Learning

This repository presents **Pyramidal Recursive learning (PyRv)**, an approach leveraging the **Pyramidal Recursive Neural Network (PyRvNN)** architecture to model text hierarchically, progressing from subwords to sentences. PyRv is designed with four key properties:
- **Hierarchical Representation**: Captures multi-level linguistic structures.
- **Representation Compositionality**: Maintains semantic integrity across levels.
- **Representation Decodability**: Ensures reconstruction of original text.
- **Self-Supervised Learning**: Learns representations without labeled data.

The implementation consists of five core modules:

| Module         | Description |
|---------------|-------------|
| `main.py`     | Orchestrates the training process, managing learning rates and depth parameters. |
| `modelarch.py` | Defines the PyRvNN architecture for hierarchical encoding and decoding. |
| `trainalg.py`  | Implements pyramidal recursion and optimization processes. |
| `dataprep.py`  | Prepares input data, including tokenization and one-hot encoding. |
| `datamanip.py` | Manages hierarchical data structures, generating representation pairs. |

If you use this work, please cite:
```
@article{babic2024recursively,
  title={Recursively Autoregressive Autoencoder for Pyramidal Text Representation},
  author={Babi{\'c}, Karlo and Me{\v{s}}trovi{\'c}, Ana},
  journal={IEEE access},
  volume={12},
  pages={71361--71370},
  year={2024},
  publisher={IEEE}
}
```
