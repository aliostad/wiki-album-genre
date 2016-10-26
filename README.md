# WikiRockWord2Vec
[![Build Status](https://travis-ci.org/aliostad/WikiRockWord2Vec.svg?branch=master)](https://travis-ci.org/aliostad/WikiRockWord2Vec)

A small Deep Learning project to illustrate the use of Deep Learning in NLP. Most of the code has been taken (and slightly modified) from [Danny Britz article](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) and its code at [github](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).

Project is Python 2.7 and uses TensorFlow to build Deep Learning model on a small dataset of Rock music album names in Wikipedia.

The dataset has been included.


## Example API
```
http://localhost:5000/api/v1/album/genre?albums=safari,soho
```

To Train:
====
```
python train.py
```

To run:
====

```
python wsgi.py
```

Make sure you change the checkpoint folder to your checkpoint.
