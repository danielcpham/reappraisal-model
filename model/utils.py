import numpy as np
from data_process import SentimentWrapper

def convert_to_wordnet(tag):
    """
    :param tag: POS tag as defined by Penn Treebank
    :return: POS tag for use in wordnet
    """
    if tag in {'NN', 'NNS', "NNP", 'NNPS', "n"}:
        return 'n'
    elif tag in {'VB', 'VBD', 'VBP', 'VBZ', 'v'}:
        return 'v'
    elif tag in {'a', 'JJ', 'JJR', 'JJS'}:
        return 'a'
    elif tag in {'r', 'RB', 'RBR', 'RBS', 'WRB'}:
        return 'r'
    else:
        return None

def normalize_sentiment(polarity, subjectivity):
    pol = convert_polarity(polarity)
    obj = convert_subj_to_obj(subjectivity)
    return SentimentWrapper(pol, obj)


def convert_polarity(pol):
    # Polarity ranges from [-1, 1] (By TextBlob API). If the polarity has absolute value 1 (-1 or 1),
    #         then set it to 0.01 to avoid getting a 0 value.
    #         Else, take the negative and add 1 to get the score between [0, 1].
    if np.abs(pol) == 1:
        return 0.01
    return -np.abs(pol) + 1


def convert_subj_to_obj(subj):
    # Subjectivity ranges from [0, 1] (By Textblob API). Subtract 1 and get absolute value such that
    #         text closer to 0 have a higher objectivity value.
    #         Else, leave it.
    if np.abs(subj) == 1:
        return 0.01
    return np.abs(subj - 1)

