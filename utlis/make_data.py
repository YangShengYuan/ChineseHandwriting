import os
import cv2
import numpy as np
import progressbar as pb

RAW_DATASET = '../data/cross/'
SAVE_DATASET = '../data/'
LABEL_FILE = '../data/label_cross.txt'

dataset_dict = {}

def save_label(file,txt_content):
    label = words_list2label_list(txt_content)
    filename = RAW_DATASET+file
    dataset_dict[filename] = label

def words_list2label_list(words):
    """
    将图像中单个文字label拼成label
    :param words:
    :return:
    """
    label_list = []

    read = open('../data/word_onehot.txt', 'rb')
    all_label_dict = read.read()
    all_label_dict = eval(all_label_dict)

    for i in words:
        if i == ' ' or i == '　':
            continue
        if i in all_label_dict.keys():
            label_list.append(all_label_dict[i])
        else:
            # 未知字符处理
            print(i)
    print(label_list)
    # f = open("asdadas/dasdad.xa")
    read.close()
    return label_list


def make_dataset():
    label_dict = {}
    for line in open(LABEL_FILE,'r'):
        picture = line.split()[0]
        label = line.split()[1]
        label_dict[picture] = label
    listdir = os.listdir(RAW_DATASET)
    pbar = pb.ProgressBar(maxval=len(listdir), widgets=['处理进度', pb.Bar('=', '[', ']'), '', pb.Percentage()])
    listdir.sort()
    i = 0
    pbar.start()
    for file in listdir:
        i += 1
        pbar.update(i)
        txt_content = label_dict[file]
        # bbox_list, words_list = extract_bbox_words(txt_content)
        # cut_img_and_save_label(file, bbox_list, words_list)
        save_label(file,txt_content)
    pbar.finish()

if __name__ == "__main__":
    # 处理train 数据集
    make_dataset()
    f = open('../data/new_crossset_label.txt', 'w')
    f.write(str(dataset_dict))
    f.close()
