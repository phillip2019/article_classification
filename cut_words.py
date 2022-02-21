# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/


# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import shutil

import jieba
import jieba.analyse  # 导入提取关䭞词的库


# 对训练集 测试集文本都进行切词处理,对测试集数据打上主题标签
# 存至文件؍
def save_file(save_path, content):
    with open(save_path, "a", encoding='utf-8', errors='ignore') as fp:
        fp.write(content)


# 读取文件
def read_file(file_path):
    with open(file_path, "r", encoding='utf-8', errors='ignore') as fp:
        content = fp.readlines()
        # print(content)
    return str(content)


# 抽取测试集的主题关键词
def extract_theme(content):
    themes = []
    tags = jieba.analyse.extract_tags(content, topK=3, withWeight=True, allowPOS=['n', 'ns', 'v', 'vn'], withFlag=True)
    for i in tags:
        # print('jieba抽取标签为: {}'.format(i))
        themes.append(i[0].word)
    return str(themes)


def cast_words(origin_path, save_path, theme_tag):
    """
    :param origin_path: 原始文本路径
    :param save_path: 切词后文本路径
    :param theme_tag: 对应文章主题关键词
    """
    file_lists = os.listdir(origin_path)  # 原文档所在路径

    # print('\n'+'file_lists:')
    # print(file_lists)
    # print('\n'+'origin_path:')
    # print(origin_path)

    # 若存在，则删除
    if theme_tag:
        shutil.rmtree(theme_tag)
        if not os.path.exists(theme_tag):
            os.makedirs(theme_tag)

    # 若存在，则删除
    if save_path:
        shutil.rmtree(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    for dir_1 in file_lists:  # 找到文件夹
        # 原始文件路径
        file_path = os.path.join(origin_path, dir_1)

        cut_path = os.path.join(save_path, dir_1)
        if not os.path.exists(cut_path):
            os.makedirs(cut_path)
        detail_paths = os.listdir(file_path)

        for detail_path in detail_paths:  # 找到文件夹下具体文件路径
            # 原始文件下每个文档路径
            full_path = os.path.join(file_path, detail_path)
            cut_save_path = os.path.join(cut_path, detail_path)
            file_content = read_file(full_path)

            file_content = file_content.strip()  # replace("\r\n", " ")
            # 删除换行
            file_content = file_content.replace("\'", "")
            file_content = file_content.replace("\\n", "")
            content_words = jieba.cut(file_content)  # 为文件内容分词
            content_words_str = " ".join(content_words)

            if theme_tag is not None:
                full_theme_path = os.path.join(theme_tag, detail_path)
                print("文件路径:{} ".format(full_theme_path))
                # theme为该文章主题关键词
                theme = extract_theme(content_words_str)
                print("文章主题关键词:{} ".format(theme))
                # 将训练集文章的主题关键词存到标签存储路径
                save_file(full_theme_path, theme)

            # print('关键词列表为: {}'.format(content_words_str))
            save_file(cut_save_path, content_words_str)  # 将处理后的文件保存到分词后语料目录


if __name__ == "__main__":
    # 对训练集进行分词
    train_words_path = 'train_segments'  # ./
    train_save_path = 'train_words'  # ./
    cast_words(train_words_path, train_save_path, theme_tag=None)
    # 对测试集进行分词 抽取文章主题标签
    test_words_path = 'test_segments'  #
    # 存放测试集文章主题标签路径  train_theme_tag/
    test_save_path = 'test_words'  #
    theme_tag_path = 'test_theme_tag'
    cast_words(test_words_path, test_save_path, theme_tag=theme_tag_path)
