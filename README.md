# Learnable Aggregator for Graph Convolutional Networks in PyTorch


<p align="center">
  <a href="https://dl.acm.org/doi/abs/10.1145/3340531.3411983"><img src="https://img.shields.io/badge/Paper-Report-red"/></a>
  <a href="https://grlearning.github.io/papers/134.pdf"><img src="https://img.shields.io/badge/Poster-NeurIPS2019-brown"/></a>
  <a href="https://github.com/asarigun/LA-GCN"><img src="https://img.shields.io/badge/TensorFlow-Implementation-ff69b4"/></a>
  <a href="https://github.com/asarigun/la-gcn-torch/blob/main/LICENSE"><img src="https://img.shields.io/github/license/thudm/cogdl"/></a>
  <a href="https://colab.research.google.com/drive/1BurTEjf6sqtIfnVn9FdCGxBwiCCHiNAN?usp=sharing" alt="license"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>


<p align="center"><img width="40%" src="https://github.com/asarigun/la-gcn-torch/blob/main/images/pytorch.png"></p>

Implementation of of Learnable Aggregators for Graph Convolutional Networks in PyTorch. 

![LA-GCN with Mask Aggregator](https://github.com/asarigun/la-gcn-torch/blob/main/images/model.jpg)


Learnable Aggregator for GCN (LA-GCN) by introducing a shared auxiliary model that provides a
customized schema in neighborhood aggregation. Under this framework, a new model proposed called
LA-GCN(Mask) consisting of a new aggregator function, mask aggregator. The auxiliary model
learns a specific mask for each neighbor of a given node, allowing both node-level and feature-level 
attention. This mechanism learns to assign different importance to both nodes and features for prediction, 
which provides interpretable explanations for prediction and increases the model robustness.

Li  Zhang ,Haiping  Lu, https://dl.acm.org/doi/abs/10.1145/3340531.3411983 (CIKM 2020) 

For official implementation  https://github.com/LiZhang-github/LA-GCN/tree/master/code


## Requirements

  * PyTorch 1.8
  * Python 3.8
  
  
## Training

```bash
python train.py
```

## Reference

[1] [Zhang & Lu, A Feature-Importance-Aware and Robust Aggregator for GCN, CIKM 2020](https://dl.acm.org/doi/abs/10.1145/3340531.3411983) [![report](https://img.shields.io/badge/Official-Code-yellow)](https://github.com/LiZhang-github/LA-GCN/tree/master/code)

[2] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907) [![report](https://img.shields.io/badge/Official-Code-ff69b4)](https://github.com/tkipf/gcn)
