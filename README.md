Pruning deep neural networks
-----

PyTorch implementation of  [1611.06440 Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)

This demonstrates pruning a VGG16 based classifier that classifies [cifar10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) dataset.


This was able to reduce the CPU runtime by x3 and the model size by x4.

For more details you can read the [blog post](https://jacobgil.github.io/deeplearning/pruning-deep-learning).

At each pruning step 512 filters are removed from the network.


Usage
-----

Training:
`python finetune.py --train`

Pruning:
`python finetune.py --prune`

TBD
---

 - Change the pruning to be done in one pass. Currently each of the 512 filters are pruned sequentually. 

  ```python
  for layer_index, filter_index in prune_targets:
  		model = prune_vgg16_conv_layer(model, layer_index, filter_index)
  ```


 	This is inefficient since allocating new layers, especially fully connected layers with lots of parameters, is slow.
	In principle this can be done in a single pass.


 - Change prune_vgg16_conv_layer to support additional architectures.
    The most immediate one would be VGG with batch norm.


Reference
---

[[jacobgil](https://github.com/jacobgil)/pytorch-pruning]