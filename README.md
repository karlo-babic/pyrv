# PyRvNN: Pyramidal Recursive Neural Network for Text Representation Learning

This repository presents **Pyramidal Recursive Learning (PyRv)**, which leverages the **Pyramidal Recursive Neural Network (PyRvNN)** to model text hierarchically, from subwords to sentences. PyRv has four main properties:
- **Representation Compositionality**: Combines multiple representations into a unified whole, capturing structured meaning.
- **Hierarchical Representation**: Represents text at multiple levels, aiding analysis of complex morphology.
- **Representation Decodability**: Enables reconstruction of original text, ensuring interpretability.
- **Self-Supervised Learning**: Learns without labeled data, unlike most recursive models requiring parse trees.

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
