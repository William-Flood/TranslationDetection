from GameManager import GameManager
from Model import Model
import tensorflow as tf
from CanvasUtils import CanvasUtils
import numpy as np
import pickle
import NLib as nl


class Controller:
    def __init__(self):
        self.IMAGE_HEIGHT = 30
        self.IMAGE_WIDTH = 30
        BOARD_SIZE = 100
        BOARD_SIGMA = 10
        self.channels = 3
        self.game_list = []
        self.game = GameManager(BOARD_SIZE, BOARD_SIGMA)
        output_size = 11
        self.model = Model(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.channels, "Fred", output_size)
        self.saved_batches = []
        self.MAX_SAVED_BATCH_SIZE = 100
        self.saved_batch_index = 1
        self.need_reset = False
        self.saved_batches_in_use = True
        self.blind_model = Model(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.channels, "George", output_size)

    @staticmethod
    def make_softmax_transform_matrix(matrix):
        flattened_labels = []
        matrix_sum = 0
        min_neg_value = 0
        for label_row in matrix:
            for label_item in label_row:
                flattened_labels.append(label_item)
                matrix_sum = matrix_sum + label_item
                if min_neg_value > label_item:
                    min_neg_value = label_item
        min_neg_value = min_neg_value * -1
        matrix_sum = matrix_sum + min_neg_value * 10
        scale = (1/matrix_sum) / (1 + (1/matrix_sum))
        min_neg_value = min_neg_value * scale
        for label_index in range(len(flattened_labels)):
            flattened_labels[label_index] = flattened_labels[label_index] * scale + min_neg_value
        flattened_labels.append(scale)
        flattened_labels.append(min_neg_value)
        return np.array(flattened_labels, np.float32)

    def make_batch(self, batch_size):
        result = None
        if self.saved_batches_in_use and self.saved_batch_index < len(self.saved_batches):
            result = self.saved_batches[self.saved_batch_index]
            self.saved_batch_index = self.saved_batch_index + 1
        else:
            if self.saved_batches_in_use:
                self.saved_batches_in_use = False
                self.saved_batches = []
            batch_inputs = []
            batch_int_outs = []
            batch_outputs = []
            for batch in range(batch_size):
                screen, labels, int_out = self.game.step()
                processed_screen = []
                for channel_index in range(len(screen)):
                    processed_channel = nl.preprocess(screen[channel_index], self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
                    processed_screen.append(processed_channel)
                current_screen_array = np.array(processed_screen, np.float32)
                current_screen_array = np.moveaxis(current_screen_array, 0, -1)
                label_array = Controller.make_softmax_transform_matrix(labels)
                int_out_softmaxed = Controller.make_softmax_transform_matrix(int_out)
                batch_outputs.append(label_array)
                batch_inputs.append(current_screen_array)
                batch_int_outs.append(int_out_softmaxed)

            result = (np.array(batch_inputs, np.float32), batch_int_outs), np.array(batch_outputs, np.float32)
            if len(self.saved_batches) < self.MAX_SAVED_BATCH_SIZE:
                self.saved_batches.append(result)
            self.need_reset = True
        return result

    def train(self):
        train_step, y_, accuracy = self.model.trainee(.00005)
        blind_train_step, blind_y_, blind_accuracy = self.blind_model.trainee(.00005)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            LOAD_EPISODE = 0
            try:
                self.model.load(LOAD_EPISODE, sess)
            except Exception as ex:
                print("Load failed")
            training_cycles = 50000
            batch_size = 200
            reset_time = 20
            batch_save_time = 10
            net_save_time = 20
            accuracy_report_frequency = 10
            save_batches = False
            max_report_cycle = 100
            max_encountered = 0
            max_encountered_on = 0
            blind_max = 0
            for i in range(training_cycles):
                print("Cycle " + str(i+1))
                if self.need_reset and i % reset_time == 0:
                    for game in self.game_list:
                        game.reset()
                inputs, outputs = self.make_batch(batch_size)
                blind_outputs = np.copy(outputs)
                np.random.shuffle(blind_outputs)
                if save_batches and self.need_reset and i != 0 and i % batch_save_time == 0 and \
                        (len(self.saved_batches) < self.MAX_SAVED_BATCH_SIZE):
                    try:
                        save_file = open("trainingData.sav", "wb")
                        archive_save_file = open("archives/trainingData_" + str(i) + ".sav", "wb")
                        pickle.dump(self.saved_batches, save_file)
                        pickle.dump(self.saved_batches, archive_save_file)
                        save_file.close()
                        archive_save_file.close()
                    except Exception as ex:
                        save_batches = False
                        print("Save error:" + str(ex))
                blind_int_outs = np.copy(inputs[1])
                np.random.shuffle(blind_int_outs)
                if (i + 1) % accuracy_report_frequency == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        self.model.x: inputs[0], self.model.int_out: inputs[1], y_: outputs, self.model.keep_prob: 1.0})
                    blind_train_accuracy = blind_accuracy.eval(feed_dict={
                        self.blind_model.x: inputs[0], self.blind_model.int_out: blind_int_outs,
                        blind_y_: blind_outputs, self.blind_model.keep_prob: 1.0})
                    print('step %d, training accuracy %g' % ((i + 1), train_accuracy))
                    print('step %d, blind training accuracy %g' % ((i + 1), blind_train_accuracy))
                    if train_accuracy > max_encountered:
                        max_encountered = train_accuracy
                        max_encountered_on = i + 1
                    if blind_train_accuracy > blind_max:
                        blind_max = blind_train_accuracy
                        max_encountered_on = i + 1
                if (i + 1) % max_report_cycle == 0:
                    print("Maximum so far: %g on step %d" % (max_encountered, max_encountered_on))
                    print("Bind Maximum so far: %g" % blind_max)
                train_step.run(feed_dict={self.model.x: inputs[0], self.model.int_out: inputs[1],
                                          y_: outputs, self.model.keep_prob: 0.5})
                blind_train_step.run(feed_dict={self.blind_model.x: inputs[0], self.blind_model.int_out: blind_int_outs,
                                                blind_y_: blind_outputs, self.blind_model.keep_prob: 0.5})
                if 0 != i and 0 == i % net_save_time:
                    self.model.save(i, sess)


if __name__ == '__main__':
    Controller().train()
