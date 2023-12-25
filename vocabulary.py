import json
import logging
import warnings
from typing import List, Set

import numpy as np
from tqdm import tqdm
import spacy


class OOVWarning(Warning):  # OutOfVocabulary警告⚠️
    pass


class EndLabelNotFoundWarning(Warning):  # 解码时到达序列长度上限，但是没有找到终止符警告⚠️
    pass


class Vocabulary:
    pad = 0
    start = 1
    end = 2
    unk = 3

    def __init__(self, json_filepath):
        with open(json_filepath, 'rb') as fp:
            self._content: dict = json.load(fp)  # str -> int
        self.inv = {v: k for k, v in self._content.items()}  # int -> str

        self.spec_words: Set[int] = {self.pad, self.start, self.end, self.unk}  # 特殊字符
        self.size = len(self._content)
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def build(raw_documents: List[str], save_path):
        """
        Vocabulary Factory
        :param raw_documents:
        :param save_path:
        :return:
        """
        content = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        nlp = spacy.load("en_core_web_sm")
        for raw_document in tqdm(raw_documents):
            for token in nlp(raw_document):
                if token.text.lower() not in content:
                    content[token.text.lower()] = len(content)
        with open(save_path, 'w') as fw:
            json.dump(content, fw)
        return Vocabulary(save_path)

    def __len__(self):
        return len(self._content)

    def __getitem__(self, word):
        if word in self._content:
            return self._content[word]
        else:
            warnings.warn(f'词表之外的词汇{word}，被编码为 <unk>', OOVWarning)
            return self.unk

    def get_word2vec(self, cache_path='word2vec.npy'):
        """获取词表的词向量嵌入, spacy提供的支持，固定提供96维词嵌入"""
        try:
            return np.load(cache_path)
        except FileNotFoundError:
            logging.log(0, 'building word2vec matrix...')
            word2vecs = []
            for idx, word in self.inv.items():
                word2vecs.append(
                    self.nlp(word)[0].tensor if idx not in self.spec_words else np.random.randn(96))  # nlp提供96维嵌入
            np.save(cache_path, np.array(word2vecs))
            return np.array(word2vecs)

    @staticmethod
    def post_process(string: str):
        s1 = string.replace(' .', '.').replace(' ,', ',').replace(' -', '-').replace('- ', '-')  # 标点间隔0
        s2 = '. '.join([s[0].upper() + s[1:] for s in s1.split('. ') if s])
        return s2

    def pad_sequence(self, indices, fixed_length):
        return (indices + [self.pad] * (fixed_length - len(indices)))[:fixed_length]

    def encode(self, sentence) -> List[int]:
        """
        将文本编码为序列
        :param sentence:
        :return:
        """
        indices = [self.start] + [self[token.text.lower()] for token in self.nlp(sentence)] + [self.end]
        return indices

    def decode(self, sequence: List[int]) -> str:
        """
        将序列解码为文本，跳过特殊字符，遇到终止符结束
        :param sequence:
        :return:
        """
        words = []
        for idx in sequence:
            if idx == self.end:
                break
            if idx not in self.inv:
                raise KeyError(f'索引{idx}超出了词表的范围，词表长度：{self.__len__()}')
            elif idx not in self.spec_words:  # 跳过特殊字符
                words.append(self.inv[idx])
        else:
            warnings.warn('没有在序列中发现终止符<end>', EndLabelNotFoundWarning)
        return self.post_process(' '.join(words))