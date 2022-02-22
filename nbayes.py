#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pickle
from sklearn.naive_bayes import MultinomialNB
import warnings
from sklearn import metrics

from cut_words import read_bunch, save_file

warnings.filterwarnings("ignore")


# 朴素贝叶斯分类
def nbayes_classify(_train_set, _test_set):
    """
    train_set: 训练集样本数据
    test_set: 测试集样本数据
    :return: 测试集样本分类
    """
    clf = MultinomialNB(alpha=0.5)
    # 训练模型
    clf.fit(_train_set.tdm, _train_set.label)
    _predict = clf.predict(_test_set.tdm)
    return _predict


def classification_result(actual, _predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, _predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, _predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, _predict, average='weighted')))


if __name__ == '__main__':
    # 导入训练集
    train_path = 'train_tfidf_space.dat'
    train_set = read_bunch(train_path)
    # 导入测试集
    test_path = "test_tfidf_space.dat"
    test_set = read_bunch(test_path)
    predict = nbayes_classify(train_set, test_set)  #
    classification_result(test_set.label, predict)
    print('-' * 100)
    # 保存结果路径
    save_path = 'classify_file.txt'
    for label, filename, predict in zip(test_set.label, test_set.filepath, predict):  # test_set
        print(filename, "\t实际类别:", label, "\t-->预测类别:", predict)
        save_content = filename + "\t实际类别:" + label + "\t-->预测类别:" + predict + '\n'
        save_file(save_path, save_content)  # 将分类结果写入txt
