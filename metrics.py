from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
import pymeteor.pymeteor as pymeteor

from vocabulary import Vocabulary

rouge = Rouge()
vocabulary = Vocabulary('vocabulary/vocab.json')


def bleu(text, standard, vocabulary):
    # 分词并转换到标签序列
    candidate = vocabulary.encode(text)
    reference = [vocabulary.encode(standard)]  # 注意：参考句子应该是一个列表的列表

    # 计算BLEU分数
    score = corpus_bleu([reference], [candidate])

    return score


def rouge_l(text, standard, vocabulary):
    score = rouge.get_scores(text, standard)[0]['rouge-l']['f']
    return score


def cidar_d(text, standard, vocabulary):
    # 分词并转换到标签序列
    candidate = text
    reference = [standard]  # 注意：参考句子应该是一个列表的列表
    # 准备CIDEr-D计算
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score({0: reference}, {0: [candidate]})

    return score


def meteor(text, standard, vocabulary):
    # 分词并转换到标签序列
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

