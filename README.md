# Learnable Aggregator for Graph Convolutional Networks in PyTorch

[![report](https://img.shields.io/badge/paper-report-red)](https://dl.acm.org/doi/abs/10.1145/3340531.3411983)  [![report](https://img.shields.io/badge/TensorFlow-Implementation-ff69b4)](https://github.com/asarigun/LA-GCN)  [![License](https://img.shields.io/github/license/thudm/cogdl)](https://github.com/asarigun/la-gcn-torch/blob/main/LICENSE)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XtLxuk0zJKxC0Ee2gMscqtAHaUIYLSH8?usp=sharing)

PyTorch implementation of of Learnable Aggregators for Graph Convolutional Networks.

![LA-GCN with Mask Aggregator](https://github.com/asarigun/LA-GCN/blob/main/model.jpg)

Learnable Aggregator for GCN (LA-GCN) by introducing a shared auxiliary model that provides a
customized schema in neighborhood aggregation. Under this framework, a new model proposed called
LA-GCN(Mask) consisting of a new aggregator function, mask aggregator. The auxiliary model
learns a specific mask for each neighbor of a given node, allowing both node-level and feature-level 
attention. This mechanism learns to assign different importance to both nodes and features for prediction, 
which provides interpretable explanations for prediction and increases the model robustness.

Li  Zhang ,Haiping  Lu, https://dl.acm.org/doi/abs/10.1145/3340531.3411983 (CIKM 2020) 

For official implementation  https://github.com/LiZhang-github/LA-GCN/tree/master/code


## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6
  
## Installation

```python setup.py install```

## Training

```bash
python train.py
```
You can also try out in colab if you don't have any requirements!  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XtLxuk0zJKxC0Ee2gMscqtAHaUIYLSH8?usp=sharing)


## Cite

Thanks for their original TensorFlow-PyTorch implementation of GCN and LA-GCN:


```
@inproceedings{
  title={A Feature-Importance-Aware and Robust Aggregator for GCN},
  author={Zhang, Li. and Lu, Haiping},
  booktitle={Conference on Information and Knowledge Management (CIKM)},
  year={2020}
}
```

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
