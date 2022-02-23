#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import io
import os
import shutil

from docx import Document
from lxml.builder import unicode


def convert_docx_to_text(_path):
    for child_dir in os.listdir(_path):
        full_child_dir = os.path.join(_path, child_dir)
        for d in os.listdir(full_child_dir):
            file_extension = d.split(".")[-1]
            if file_extension == "docx" or file_extension == 'doc':
                docx_filename = os.path.join(full_child_dir, d)
                document = Document(docx_filename)
                text_filename = os.path.join(full_child_dir, d.split(".")[0] + ".txt")
                if os.path.exists(text_filename):
                    os.remove(text_filename)

                with io.open(text_filename, "a+", encoding="utf-8") as fp:
                    for para in document.paragraphs:
                        fp.write(unicode(para.text))

                # 若写入完成，则删除原始文件
                if os.path.exists(docx_filename):
                    os.remove(docx_filename)


if __name__ == '__main__':
    path = r"train_segments"
    convert_docx_to_text(path)