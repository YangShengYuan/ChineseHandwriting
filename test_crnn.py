import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from crnn import CRNN
from utlis.net_cfg_parser import parser_cfg_file
import os

result_list = []


class Test_CRNN(object):
    def __init__(self, batch_size=None):
        net_params, train_params = parser_cfg_file('./net.cfg')
        self._model_save_path = str(net_params['model_load_path'])
        self.input_img_height = int(net_params['input_height'])
        self.input_img_width = int(net_params['input_width'])
        if batch_size is None:
            self.test_batch_size = int(net_params['test_batch_size'])
        else:
            self.test_batch_size = batch_size

        # 加载label onehot
        f = open('./data/word_onehot.txt', 'r', encoding='utf-8')
        data = f.read()
        words_onehot_dict = eval(data)
        self.words_list = list(words_onehot_dict.keys())
        self.words_onehot_list = [words_onehot_dict[self.words_list[i]] for i in range(len(self.words_list))]

        # 构建网络
        self.inputs_tensor = tf.placeholder(tf.float32,
                                            [self.test_batch_size, self.input_img_height, self.input_img_width, 1])
        self.seq_len_tensor = tf.placeholder(tf.int32, [None], name='seq_len')

        crnn_net = CRNN(net_params, self.inputs_tensor, self.seq_len_tensor, self.test_batch_size, True)
        net_output, decoded, self.max_char_count = crnn_net.construct_graph()
        self.dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, "./model/ckpt")

    def _get_input_img(self, img_path_list):

        batch_size = len(img_path_list)

        batch_data = np.zeros([batch_size,
                               self.input_img_height,
                               self.input_img_width,
                               1])
        img_list = []

        for i in range(batch_size):
            img = cv2.imread(img_path_list[i], 0)
            img_list.append(img)
            # print(np.shape(img))
            # print(img_path_list[i])
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = self._resize_img(img)
            reshape_img = resized_img.reshape([1, self.input_img_height, self.input_img_width, 1])
            img_norm = reshape_img / 255 * 2 - 1
            batch_data[i] = img_norm

        return batch_data, batch_size, img_list

    def test_img(self, img_path_list, is_show_res=False):

        batch_data, batch_size, img_list = self._get_input_img(img_path_list)
        if batch_size != self.test_batch_size:
            error = '网络构建batch size:' + str(self.test_batch_size) + '和实际输入batch size:' + str(batch_size) + '不一样'
            assert 0, error

        feed_dict = {self.inputs_tensor: batch_data, self.seq_len_tensor: [self.max_char_count] * batch_size}
        predict = self.sess.run(self.dense_decoded, feed_dict=feed_dict)
        predict_seq = self._predict_to_words(predict)

        if is_show_res:
            for i in range(batch_size):
                result = img_path_list[i].split('/')[-1] + ' ' + predict_seq[i]
                print(result)
                result_list.append(result)
                # cv2.imshow(img_path_list[i], img_list[i])
            # cv2.waitKey()

        return predict_seq

    def _predict_to_words(self, decoded):
        words = []

        for seq in decoded:
            seq_words = ''
            for onehot in seq:
                if onehot == -1:
                    break
                if onehot not in self.words_onehot_list:
                    seq_words += ' '
                else:
                    seq_words += self.words_list[self.words_onehot_list.index(onehot)]
            words.append(seq_words)
        return words

    def _resize_img(self, img):
        """
        将图像先转为灰度图，并将图像进行resize
        :param img:
        :return:
        """
        height, width = np.shape(img)

        if width > self.input_img_width:
            width = self.input_img_width
            ratio = float(self.input_img_width) / width
            outout_img = cv2.resize(img, (self.input_img_width, self.input_img_height))
        else:
            outout_img = np.zeros([self.input_img_height, self.input_img_width])
            ratio = self.input_img_height / height
            img_resized = cv2.resize(img, (int(width * ratio), self.input_img_height))
            outout_img[:, 0:np.shape(img_resized)[1]] = img_resized

        return outout_img


if __name__ == "__main__":
    # batch size = 16 batch numbers =142
    test_img_list = []
    test_file_path = './test/'
    listdir = os.listdir(test_file_path)
    one_batch_list = []
    i = 0
    for file in listdir:
        one_batch_list.append(test_file_path + file)
        i += 1
        if i % 16 == 0:
            i = 0
            test_img_list.append(one_batch_list)
            one_batch_list = []
    print(test_img_list)

    a = Test_CRNN()
    for one_batch in test_img_list:
        a.test_img(one_batch, is_show_res=True)

    file = open("char_test.txt", "w")
    for item in result_list:
        file.write(item + "\n")
    file.close()
