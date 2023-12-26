"""
Evaluation Metrics for Image Captioning

This module provides implementation for various evaluation metrics commonly
used in image captioning and other natural language processing tasks. It includes
methods for calculating BLEU, ROUGE and METEOR scores. These metrics are
essential for evaluating the quality of generated captions compared to reference
captions.

The module uses external libraries such as nltk and rouge
to facilitate the computation of these metrics. A custom Vocabulary class is used
for encoding and decoding text.

Author: zhanghanmo2021213368, hammershock
Date: 2023/10/26
License: MIT License
Usage: This file is part of a larger project FashionDescription.
It is not intended to be used independently.

Dependencies:
- nltk: for BLEU score calculation
- rouge: for ROUGE score calculation
"""

from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu

from vocabulary import Vocabulary

__rouge = Rouge()


def bleu(text, standard, vocabulary):
    """计算BLEU指标"""
    candidate = vocabulary.encode(text)
    reference = [vocabulary.encode(standard)]  # 注意：参考句子应该是一个列表的列表
    score = corpus_bleu([reference], [candidate])
    return score


def rouge_l(text, standard, vocabulary):
    """计算Rouge-l指标"""
    score = __rouge.get_scores(text, standard)[0]['rouge-l']['f']
    return score


def _find_chunks(candidate, reference):
    """寻找chunks（连续匹配单词序列）"""
    candidate_chunks = []
    reference_chunks = []
    chunk = []

    for word in candidate:
        if word in reference:
            if not chunk:
                chunk_start = reference.index(word)
            chunk.append(word)
        else:
            if chunk:
                candidate_chunks.append(chunk)
                reference_chunks.append(reference[chunk_start:chunk_start+len(chunk)])
                chunk = []

    if chunk:  # 处理最后一个chunk
        candidate_chunks.append(chunk)
        reference_chunks.append(reference[chunk_start:chunk_start+len(chunk)])

    return candidate_chunks, reference_chunks


def meteor(candidate, reference, vocabulary):
    """一个简化版本的METEOR算法实现，没有考虑同义词匹配和词根匹配"""
    candidate_words = vocabulary.split(candidate)
    reference_words = vocabulary.split(reference)

    candidate_chunks, reference_chunks = _find_chunks(candidate_words, reference_words)

    # 计算匹配数和chunk数
    matches = sum(len(chunk) for chunk in candidate_chunks)
    num_chunks = len(candidate_chunks)

    # 计算精确率和召回率
    precision = matches / len(candidate_words)
    recall = matches / len(reference_words)

    f_score = 0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    # 计算惩罚因子
    penalty = 0.5 * ((num_chunks / matches) ** 3) if matches != 0 else 0

    # 计算最终的METEOR分数
    meteor = f_score * (1 - penalty)

    return meteor


if __name__ == "__main__":
    vocabulary = Vocabulary('vocabulary/vocab.json')
    # 以下是一个简单的演示示例
    standard = ("The person is wearing a tank tank shirt with pure color patterns. The tank shirt is with cotton "
                "fabric. It has a round neckline. The pants this lady wears is of long length. The pants are with "
                "cotton fabric and pure color patterns. There is an accessory on her wrist. This female wears a ring.")

    text = ("This female is wearing a long-sleeve shirt with solid color patterns. The shirt is with knitting fabric "
            "and its neckline is round. The trousers this female wears is of long length. The trousers are with denim "
            "fabric and solid color patterns. This person is wearing a ring on her finger. There is clothing on her "
            "waist.")

    print(rouge_l(text, standard, vocabulary))
    print(bleu(text, standard, vocabulary))
    print(meteor(text, standard, vocabulary))

