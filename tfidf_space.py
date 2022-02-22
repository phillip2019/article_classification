#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.datasets._base import Bunch


# 训练集
from cut_words import read_bunch, read_file, write_bunch


def train_tfidf_space(_stop_word_path, _train_bunch_path, _train_tfidf_data):
    """
    stopword_path: 停用词路径
    train_bunch_path: 训练集语料路径
    train_tfidf_data: 训练集tfidf数据路径
    """
    bunch = read_bunch(_train_bunch_path)
    # 读取停用词
    stop_words = read_file(_stop_word_path).splitlines()
    tfidf_space = Bunch(label=bunch.label, filepath=bunch.filepath, contents=bunch.contents, tdm=[], space={})
    # max_df是最大文件数，当为int格式时即最大几份，当为float格式时即占比多少份，min_df默认为int 1即一份
    vectorizer = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True, max_df=0.5)
    # TF-IDF权重矩阵matrix
    tfidf_space.tdm = vectorizer.fit_transform(bunch.contents)
    # vocabulary_属性表示生成的词汇表以及每个单词对应的编码（也可以认为是index）
    tfidf_space.space = vectorizer.vocabulary_
    write_bunch(_train_tfidf_data, tfidf_space)


# 测试集
def test_tfidf_space(_stop_word_path, _test_bunch_path, _test_tfidf_data, _train_tfidf_data):
    """
    _stop_word_path: 停用词路径
    _test_bunch_path: 测试集语料路径
    _test_tfidf_data: 测试集tfidf数据路径
    _train_tfidf_data: 训练集tfidf数据路径,将训练集的词向量空间坐标赋值给测试集
    """
    bunch = read_bunch(_test_bunch_path)
    stopwords = read_file(_stop_word_path).splitlines()  # 读取停用词
    tfidf_space = Bunch(label=bunch.label, filepath=bunch.filepath, contents=bunch.contents, tdm=[], space={})
    # 训练集tfidf数据
    train_bunch = read_bunch(_train_tfidf_data)
    # 将训练集的词向量空间坐标赋值给测试集
    tfidf_space.space = train_bunch.space
    # max_df是最大文件数，当为int格式时即最大几份，当为float格式时即占比多少份，min_df默认为int 1即一份
    vectorizer = TfidfVectorizer(stop_words=stopwords, sublinear_tf=True, max_df=0.5,
                                 vocabulary=train_bunch.space)
    tfidf_space.tdm = vectorizer.fit_transform(bunch.contents)
    write_bunch(_test_tfidf_data, tfidf_space)


if __name__ == '__main__':
    # 训练集数据处理
    stop_word_path = "chinese_stop_words.txt"  # 用词表的路径
    train_bunch_path = 'train_bunch_bag.dat'
    train_tfidf_data = 'train_tfidf_space.dat'
    train_tfidf_space(stop_word_path, train_bunch_path, train_tfidf_data)
    # 测试集数据处理
    test_bunch_path = 'test_bunch_bag.dat'
    test_tfidf_data = 'test_tfidf_space.dat'
    test_tfidf_space(stop_word_path, test_bunch_path, test_tfidf_data, train_tfidf_data)
