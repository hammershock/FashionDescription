"""
Evaluation Metrics for Image Captioning

This module provides implementation for various evaluation metrics commonly
used in image captioning and other natural language processing tasks. It includes
methods for calculating BLEU, ROUGE, CIDEr, and METEOR scores. These metrics are
essential for evaluating the quality of generated captions compared to reference
captions.

The module uses external libraries such as nltk, rouge, pycocoevalcap, and pymeteor
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
- pycocoevalcap: for CIDEr score calculation
- pymeteor: for METEOR score calculation
"""


from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
import pymeteor.pymeteor as pymeteor

from vocabulary import Vocabulary

rouge = Rouge()
vocabulary = Vocabulary('vocabulary/vocab.json')
cider_scorer = Cider()


def bleu(text, standard, vocabulary):
    candidate = vocabulary.encode(text)
    reference = [vocabulary.encode(standard)]  # 注意：参考句子应该是一个列表的列表
    score = corpus_bleu([reference], [candidate])
    return score


def rouge_l(text, standard, vocabulary):
    score = rouge.get_scores(text, standard)[0]['rouge-l']['f']
    return score


def cidar_d(text, standard, vocabulary):
    candidate = text
    reference = [standard]  # 注意：参考句子应该是一个列表的列表
    score, _ = cider_scorer.compute_score({0: reference}, {0: [candidate]})
    return score


def meteor(text, standard, vocabulary):
    candidate = text  # 这里需要文本形式，而不是索引
    reference = standard.replace('.', ' ').replace(',', ' ').replace('-', ' ')
    score = pymeteor.meteor(reference, candidate)
    return score


if __name__ == "__main__":
    standard = ("The person is wearing a tank tank shirt with pure color patterns. The tank shirt is with cotton "
                "fabric. It has a round neckline. The pants this lady wears is of long length. The pants are with "
                "cotton fabric and pure color patterns. There is an accessory on her wrist. This female wears a ring.")
    text = ("This female is wearing a long-sleeve shirt with solid color patterns. The shirt is with knitting fabric "
            "and its neckline is round. The trousers this female wears is of long length. The trousers are with denim "
            "fabric and solid color patterns. This person is wearing a ring on her finger. There is clothing on her "
            "waist.")

    print(rouge_l(text, standard, vocabulary))
    print(bleu(text, standard, vocabulary))

