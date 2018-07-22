import tensorflow as tf
import numpy as np
import os
import re
import NLib as nl


class Model:
    def __init__(self, height, width, channels, model_name, output_size):
        self.output_size = output_size
        self.model_name = model_name
        self.x = tf.placeholder(
            tf.float32, [None, height, width, channels], name=(model_name + '-' + 'X'))
        self.conv_1 = nl.conv2d_layer(self.x, 8, 64, model_name + '-' + 'conv1', strides=[1, 1, 1, 1])
        self.conv_2 = nl.conv2d_layer(self.conv_1, 8, 64, model_name + '-' + 'conv2', strides=[1, 1, 1, 1])
        self.pool_2 = Model.max_pool_2x2(self.conv_2, "pool2")
        self.conv_3 = nl.conv2d_layer(self.pool_2, 8, 256, model_name + '-' + 'conv3', strides=[1, 4, 4, 1])
        with tf.name_scope(model_name + '-' + "flattened"):
            self.conv_3_flattened = tf.reshape(self.conv_3, [-1, Model.total_size(self.conv_3)])

        self.int_out = tf.placeholder(
            tf.float32, [None, output_size], name= model_name + '-' + 'INT_OUT')
        with tf.name_scope(model_name + '-' + "flattened_and_int_out"):
            self.fully_connected_feed = tf.concat([self.conv_3_flattened, self.int_out], 1)
        self.fully_connected_1 = nl.dense_layer(self.fully_connected_feed, 512, model_name + '-' + 'fc1')
        self.fully_connected_2 = nl.dense_layer(self.fully_connected_1, 512, model_name + '-' + 'fc2')
        self.keep_prob = None
        self.fc_drop = self.dropout(self.fully_connected_2, model_name + '-' + "dropout_1")
        self.output = nl.dense_layer(self.fc_drop, output_size, model_name + '-' + 'fc1o', None)
        tf_vars = tf.global_variables()
        self.saver = tf.train.Saver({var.name: var for var in tf_vars}, max_to_keep=0)

    @staticmethod
    def max_pool_2x2(x, name):
        """max_pool_2x2 downsamples a feature map by 2X."""
        with tf.variable_scope(name):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

    def dropout(self, input_tensor, name):
        with tf.name_scope(name):
            self.keep_prob = tf.placeholder(tf.float32)
            return tf.nn.dropout(input_tensor, self.keep_prob)

    @staticmethod
    def total_size(input_tensor):
        size = 1
        for dimension in input_tensor.get_shape().as_list():
            if dimension is not None:
                size = size * dimension
        return size

    def trainee(self, learning_rate):
        y_ = tf.placeholder(tf.float32, [None, self.output_size])
        with tf.name_scope(self.model_name + '-' + 'loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                    logits=self.output)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope(self.model_name + '-' + 'adam_optimizer'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        
        with tf.name_scope(self.model_name + '-' + 'accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        return train_step, y_, accuracy

    # def _checkpoint_filename(self, episode):
    #     return 'checkpoints/%s_%08d' % (self.model_name, episode)

    # def _get_episode_from_filename(self, filename):
    #     # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
    #     split_name = re.split('/|_|\.', filename)
    #     return int(split_name[1])

    def save(self, episode, session):
        return nl.save(self, episode, session)
        # self.saver.save(session, self._checkpoint_filename(episode))

    def load(self, episode, session):
        return nl.load(self, episode, session)
        # filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
        # if episode > 0:
        #     filename = self._checkpoint_filename(episode)
        # self.saver.restore(session, filename)
        # return self._get_episode_from_filename(filename)

