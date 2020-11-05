# TensorFlow Recommenders-Addons

![TensorFlow Recommenders-Addons logo](assets/SIGRecommendersAddons.png)

![TensorFlow Recommenders build badge](https://github.com/tensorflow/recommenders/workflows/TensorFlow%20Recommenders/badge.svg)
[![PyPI badge](https://img.shields.io/pypi/v/tensorflow-recommenders-addons.svg)](https://pypi.python.org/pypi/tensorflow-recommenders-addons/)

TensorFlow Recommenders Addons is a library for building recommender system models
using [TensorFlow](https://www.tensorflow.org).

It makes TensorFlow support dynamic sparse weights training for large scale recommendation systems (Recommenders), which are one of most common and impactful use cases in the industry. We hope to encourage sharing of best practices in the industry, get consensus and product feedback to help evolve TensorFlow better, and facilitate the contributions of RFCs and PRs in this domain.

## Installation

Make sure you have TensorFlow 2.x installed, and install from `pip`:

```shell
pip install tensorflow-recommenders-addons

```

## Documentation

Have a look at our
[tutorials](https://tensorflow.org/recommenders/examples/quickstart) and
[API reference](https://www.tensorflow.org/recommenders-addons/api_docs/python/tfra/).

## Quick start

Building a factorization model for the Movielens 100K dataset is very simple
([Colab](https://tensorflow.org/recommenders-addons/examples/quickstart)):

```python
import tensorflow as tf
import tensorflow_recommenders_addons as tfra

# graph defination
x, labels = dataset
w = tfra.get_variable(name="dynamic_embeddings",
                      devices=[
                          "/job:ps/replica:0/task:0/CPU:0",
                          "/job:ps/replica:0/task:1/CPU:0"
                      ],
                      initializer=tf.random_normal_initializer(0, 0.005),
                      dim=1)
z = tfra.embedding_lookup(params=w, ids=x, name="wide-sparse-weights")

# graph defination
opt = tfra.DynamicEmbeddingOptimizer(tf.train.AdamOptimizer(0.001))
update = opt.minimize(loss)
saver = tf.train.Saver()

# training loop
with tf.Session() as sess:
  for _ in range(100):
    sess.run(update)
  saver.save(sess, './ckpt/ckpt')

```
