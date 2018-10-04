# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import tensorflow as tf
import numpy as np
import re
import os
import scipy.misc as misc

def conv2d_layer(input_tensor, filter_size, out_dim, name, strides, func=tf.nn.relu):
    in_dim = input_tensor.get_shape().as_list()[-1]
    d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)
        w = tf.get_variable('w',
                            shape=[filter_size, filter_size, in_dim, out_dim],
                            dtype=tf.float32,
                            initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

        output = tf.nn.conv2d(input_tensor, w, strides=strides, padding='SAME') + b
        if func is not None:
            output = func(output)
    return output, w, b


def dense_layer(input_tensor, out_dim, name, func=tf.nn.relu):
    in_dim = input_tensor.get_shape().as_list()[-1]
    d = 1.0 / np.sqrt(in_dim)
    with tf.variable_scope(name):
        w_init = tf.random_uniform_initializer(-d, d)
        b_init = tf.random_uniform_initializer(-d, d)
        w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
        b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

        output = tf.matmul(input_tensor, w) + b
        if func is not None:
            output = func(output)
    return output, w, b


def _checkpoint_filename(self, episode):
    return 'checkpoints/%s_%08d' % (self.model_name, episode)


def _get_episode_from_filename(filename):
    # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
    split_name = re.split('/|_|\.', filename)
    return int(split_name[1])


def save(model, episode, session):
    model.saver.save(session, _checkpoint_filename(model, episode))


def load(model, episode, session):
    filename = tf.train.latest_checkpoint(os.path.dirname(_checkpoint_filename(model, episode=0)))
    if episode > 0:
        filename = _checkpoint_filename(model, episode)
    model.saver.restore(session, filename)
    return _get_episode_from_filename(filename)


def _rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def preprocess(image, image_height, image_width):
    image = _rgb2gray(image)
    image = misc.imresize(image, [image_height, image_width], 'bilinear')
    image = image.astype(np.float32) / 128.0 - 1.0
    return image
