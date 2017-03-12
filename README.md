# HandwritingRecognition (MultiDimensional RecurrentNeuralNetworks)
Recognize handwritten text in scanned documents using MultiDimensional Recurrent Neural Networks

Creates a network based on [MultiDimensional RNNs](http://people.idsia.ch/~juergen/nips2009.pdf) architecture using python and cython with Connectionist Temporal Classification (CTC) cost function.

## Features
* Created Multidimensional LSTM network.
* No need to extract features before feeding it to RNN or LSTM framework.
* The current configuration takes 2D input but can be extended to N-dimensional input.
* Uses forward backward algorithm with CTC loss function. This is taken from Andrew Mass [stanford-ctc] (https://github.com/amaas/stanford-ctc).
* Runs on Multi-Cores.
* Uses cython for fast execution.

## Installation
* Installing cython from [Cython] (http://cython.readthedocs.io/en/latest/src/quickstart/install.html)
* Installing dill from [dill] (https://pypi.python.org/pypi/dill). dill extends python’s pickle module for serializing and de-serializing python objects.
