# LA-GCN 

[![report](https://img.shields.io/badge/paper-report-red)](https://dl.acm.org/doi/abs/10.1145/3340531.3411983)  [![report](https://img.shields.io/badge/official-code-ff69b4)](https://github.com/LiZhang-github/LA-GCN/tree/master/code)  [![License](https://img.shields.io/github/license/thudm/cogdl)](https://github.com/asarigun/LA-GCN/blob/main/LICENSE)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XtLxuk0zJKxC0Ee2gMscqtAHaUIYLSH8?usp=sharing)

This repository is tensorflow implementation of Learnable Aggregators for Graph Convolutional Networks.

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
* tensorflow_version 1.x

## Training

```bash
python train.py
```
You can also try out in colab if you don't have any requirements!  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XtLxuk0zJKxC0Ee2gMscqtAHaUIYLSH8?usp=sharing)

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), 
* an N by D feature matrix (D is the number of features per node), and
* an N by E binary label matrix (E is the number of classes).

Have a look at the `load_data()` function in `utils.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://www.cs.umd.edu/~sen/lbc-proj/LBC.html. In our version (see `data` folder) we use dataset splits provided by https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861), ICML 2016). 

You can specify a dataset as follows:

```bash
python train.py --dataset citeseer
```

(or by editing `train.py`)

## Models

You can choose between the following models: 
* `gcn`: Graph convolutional network (Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907), 2016)
* `gcn_cheby`: Chebyshev polynomial version of graph convolutional network as described in (Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), NIPS 2016)
* `dense`: Basic multi-layer perceptron that supports sparse inputs

## Cite

Thanks for their original tensorflow implementation of GCN and LA-GCN:

```
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N. and Welling, Max},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}
```

```
@inproceedings{
  title={A Feature-Importance-Aware and Robust Aggregator for GCN},
  author={Zhang, Li. and Lu, Haiping},
  booktitle={Conference on Information and Knowledge Management (CIKM)},
  year={2020}
}
```





Graph Convolutional Networks in PyTorch
====

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

For a high-level introduction to GCNs, see:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

![Graph Convolutional Networks](figure.png)

Note: There are subtle differences between the TensorFlow implementation in https://github.com/tkipf/gcn and this PyTorch re-implementation. This re-implementation serves as a proof of concept and is not intended for reproduction of the results reported in [1].

This implementation makes use of the Cora dataset from [2].

## Installation

```python setup.py install```

## Requirements

  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6

## Usage

```python train.py```

## References

[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)

[2] [Sen et al., Collective Classification in Network Data, AI Magazine 2008](http://linqs.cs.umd.edu/projects/projects/lbc/)

## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
```
