# caloGraphNN

Repository that contains minimal implementations of the graph neural network layers discussed in [arxiv:xxxx].
All necessary functions are included in the individual python files. The layers can be used analogously to tensorflow layers. 
Keras wrappers will follow.
The bare layers can be found in caloGraphNN.py, and can be used in a similar way as bare tensorflow layers, and therefore can be easily implemented in custom DNN architectures.
The source code for models described in the paper is in tensorflow_models.py for reference.

The tensorflow version needs to be at least 1.8.

When using these layers to build models or modifying them, please cite our paper:

```
@article{caloGraphNN,
  author    = {Qasim, Shah Rukh and Kieseler, Jan and Iiyama, Yutaro and Pierini, Maurizio},
  title     = {Learning representations of irregular particle-detector geometry with distance-weighted graph networks},
  journal   = {XX},
  volume    = {abs/xxx},
  year      = {2019},
  url       = {https://arxiv.org/abs/1902.07987},
  archivePrefix = {arXiv},
  eprint    = {1902.07987},
}
```
