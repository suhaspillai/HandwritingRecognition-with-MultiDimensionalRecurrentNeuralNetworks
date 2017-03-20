# HandwritingRecognition (MultiDimensional RecurrentNeuralNetworks)
Recognize handwritten text in scanned documents using MultiDimensional Recurrent Neural Networks

Creates a network based on [MultiDimensional RNNs](http://people.idsia.ch/~juergen/nips2009.pdf) architecture using python and cython with Connectionist Temporal Classification (CTC) cost function.

## Features
* Creates Multidimensional LSTM network.
* No need to extract features before feeding it to RNN or LSTM framework.
* The current configuration takes 2D input but can be extended to N-dimensional input.
* Uses forward backward algorithm with CTC loss function. This is taken from Andrew Mass [stanford-ctc] (https://github.com/amaas/stanford-ctc).
* Runs on Multi-Cores.
* Uses cython for fast execution.

## Installation
* Installing cython from [Cython](http://cython.readthedocs.io/en/latest/src/quickstart/install.html)
* Installing dill from [dill](https://pypi.python.org/pypi/dill). dill extends python’s pickle module for serializing and de-serializing python objects.

## Data Preparation
Downloading IAM dataset for handwriting recognition from [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database).
To create data splits for training, validation and testing 

```
python create_data.py path_to_xml_files path_to_words train_samples val_samples 
```
path_to_xml_files: folder where xml files are stored.
path_to_words: folder where images of handwritten words are stored.
train_samples: no of training samples
val_samples: no of validation samples

The IAM dataset contains 115149 images of words, so the following command will create 80k trainig_data 15k validation_data and 20k testing_data. 
```
python create_data.py /home/xml_files /home/data/words 80000 15000
````

