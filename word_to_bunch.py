#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import pickle

from sklearn.utils import Bunch

'''
label: 文章类型
file_path: 文章路径
contents: 分词后的文章
'''


def read_file(file_path):
    with open(file_path, "r", encoding='utf-8', errors='ignore') as fp:
        content = fp.readlines()
    return str(content)


def word_to_bunch(_train_save_path, _train_bunch_path):
    bunch = Bunch(label=[], filepath=[], contents=[])

    all_labels = os.listdir(_train_save_path)
    for label in all_labels:
        detail_path = os.path.join(_train_save_path, label)

        all_details = os.listdir(detail_path)
        for all_detail in all_details:
            # 文件具体路径
            file_detail_path = os.path.join(detail_path, all_detail)
            bunch.label.append(label)
            bunch.filepath.append(file_detail_path)
            contents = read_file(file_detail_path)
            bunch.contents.append(contents)
    with open(_train_bunch_path, "wb+") as fp:
        pickle.dump(bunch, fp)
    print("创建完成")


if __name__ == "__main__":
    train_save_path = 'train_words'
    train_bunch_path = "train_bunch_bag.dat"
    word_to_bunch(train_save_path, train_bunch_path)
    test_save_path = 'test_words'
    test_bunch_path = "test_bunch_bag.dat"
    word_to_bunch(test_save_path, test_bunch_path)
