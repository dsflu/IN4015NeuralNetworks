# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(0.01))
    # print output.get_shape()[1]
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-5,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


class MLPModel3Layers(models.BaseModel):

  def create_model(self, model_input, vocab_size, **unused_params):
      
    net = slim.fully_connected(model_input,1024)
    net = slim.fully_connected(model_input,2048)
    net = slim.fully_connected(net,4096)
    dropout = tf.layers.dropout(
      inputs=net, rate=0.5)

    output = slim.fully_connected(
    dropout, vocab_size, activation_fn=tf.nn.sigmoid,
    weights_regularizer=slim.l2_regularizer(0.01))

    return {"predictions": output}

class MLPModeltest(models.BaseModel):

  def create_model(self, model_input, vocab_size, **unused_params):
      
    # network = tl.layers.InputLayer(model_input, name='input_layer')
    # network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    # network = tl.layers.DenseLayer(network, n_units=1024, act = tf.nn.relu, name='relu1')
    # network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    # network = tl.layers.DenseLayer(network, n_units=1024, act = tf.nn.relu, name='relu2')
    # network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    input_layer = slim.fully_connected(model_input, 4000, activation_fn=tf.nn.relu)
    hidden_layer1 = slim.fully_connected(input_layer, 8000, activation_fn=tf.nn.relu)
    hidden_layer2 = slim.fully_connected(hidden_layer1, 5000, activation_fn=tf.nn.relu)
    # output = slim.fully_connected(hidden_layer, vocab_size, activation_fn=tf.nn.softmax)

    output = slim.fully_connected(
    hidden_layer2, vocab_size, activation_fn=tf.nn.sigmoid,
    weights_regularizer=slim.l2_regularizer(0.01))

    return {"predictions": output}




class CNNModel2(models.BaseModel):

  def create_model(self, model_input, vocab_size, **unused_params):
    print model_input.get_shape()[1]
    print vocab_size
      
    
    input_layer = tf.reshape(model_input, [-1,32,32,1])
    
    
    net = slim.conv2d(input_layer, 10, [3, 3])

    
    net = slim.max_pool2d(net, [32,32], [32,32], padding="same")   

    output = slim.fully_connected(
    net, vocab_size, activation_fn=tf.nn.sigmoid,
    weights_regularizer=slim.l2_regularizer(0.01))

    print output.get_shape()[0]
    print output.get_shape()[1]

    return {"predictions": output}

class CNNModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):

    # Input Layer
    input_layer = tf.reshape(model_input, [-1, 32, 32, 1])


    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=[7, 7],
      strides=(2, 2),
      padding="same",
      activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[5, 5],
      strides=(2, 2),
      padding="same",
      activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    pool3_flat = tf.reshape(pool3, [-1, 524288])
    dense = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4)
    dense2 = tf.layers.dense(inputs=dropout, units=2048, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.4)

    # logits = tf.layers.dense(inputs=dropout2, units=10)


    output = slim.fully_connected(
        dropout2, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    print output.get_shape()[1]
    return {"predictions": output}

