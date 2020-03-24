from abc import ABCMeta
from dataclasses import dataclass
import pandas as pd
import pickle
import os

farAwayFlag = 'spatiotemp'
objectiveFlag = 'obj'

def data_partition(data: pd.DataFrame, frac):
    # Split data into training data and testing data.
    data_train = data.sample(frac=frac)
    # Takes test data to be the difference of the full data and the training data.
    data_test = data[~data.apply(tuple, 1).isin(
        data_train.apply(tuple, 1))]
    return data_train, data_test

def read_liwc_dictionary(filename):
    """
    :param filename: 
    :return: wordbank
    """
    wordbank = {}
    df = pd.read_excel(open(filename, 'rb'))
    for column_name in df.columns:
        wordbank[column_name] = list(df[column_name].dropna())

    return wordbank

cwd = os.getcwd()
wordbank = read_liwc_dictionary(cwd + "/input/data/WordBank - LIWC2007.xlsx")



class Strategy(metaclass= ABCMeta):
    """
    Defines a common class for Reappraisal Strategies.
    """
    name = "generic"
    categories = set([])
    posCategories = set([])
    negCategories = set([])


    def classifier(self, word, tag):
        pass

class SpatioTempStrategy(Strategy):
    name = farAwayFlag
    discrep = wordbank['Discrepancy Words']
    pos_space = wordbank['Pos Space']
    neg_space = wordbank['Neg Space']
    pos_time = wordbank['Pos Time']
    neg_time = wordbank['Neg Time']
    pos_articles = wordbank['Pos Far Articles']
    neg_articles = wordbank['Neg Far Articles']

    def __init__(self):
        self.categories = ['discrep', 'pos_space', 'neg_space', 'pos_time', 'neg_time', "fps",
              'past_tense', 'present_tense', 'future_tense', 'greater_than_6', 'pos_articles', 'neg_articles']
        self.posCategories = {'discrep', 'pos_space', 'pos_time', 'past_tense','future tense', 'greater_than_6', "pos_articles"}
        self.negCategories = {'neg_space', 'neg_time', 'fps', 'present_tense', 'neg_articles'}

    def classifier(self, word, tag):
        """
            :param word, tag: word, POS-tag pairing
            :return: a list of all categories it belongs to 
            """

        category_match = []
        if tag in {'PRP', "PRP$"}:
            category_match.append('fps')
        elif tag in {'VB', 'VBP'}:
            category_match.append('present_tense')
        elif tag == 'MD' and word == 'will':
            category_match.append('future_tense')
        elif tag == 'VBD':
            category_match.append('past_tense')

        if word in self.discrep:
            category_match.append('discrep')

        if word in self.pos_space:
            category_match.append('pos_space')
        elif word in self.neg_space:
            category_match.append('neg_space')

        if word in self.pos_time:
            category_match.append('pos_time')
        elif word in self.neg_time:
            category_match.append('neg_time')

        if word in self.pos_articles:
            category_match.append('pos_articles')
        elif word in self.neg_articles:
            category_match.append('neg_articles')

        if len(word) > 6:
            category_match.append('greater_than_6')
        return category_match


class ObjectiveStrategy(Strategy):
    name = objectiveFlag
    discrep = wordbank['Discrepancy Words']
    pos_articles = wordbank['Pos Obj Articles']
    neg_articles = wordbank['Neg Obj Articles']
    certainty = wordbank['Certainty']


    def __init__(self):
        self.categories = ['discrep', "fps", 'certainty',
                  'past_tense', 'present_tense', 'future_tense', 'greater_than_6', 'pos_articles', 'neg_articles']
        self.posCategories = {'present_tense', 'greater_than_6', 'pos_articles', 'certainty'}
        self.negCategories = {'discrep', 'fps', 'past_tense', 'future_tense', 'neg_articles'}

    def classifier(self, word, tag):
        """
            :param word, tag: word, POS-tag pairing
            :return: a list of all categories it belongs to 
            """
        category_match = []
        if tag in {'PRP', "PRP$"}:
            category_match.append('fps')
        elif tag in {'VB', 'VBP'}:
            category_match.append('present_tense')
        elif tag == 'MD' and word == 'will':
            category_match.append('future_tense')
        elif tag == 'VBD':
            category_match.append('past_tense')

        if word in self.discrep:
            category_match.append('discrep')

        if word in self.certainty:
            category_match.append('certainty')

        if word in self.pos_articles:
            category_match.append('pos_articles')
        elif word in self.neg_articles:
            category_match.append('neg_articles')

        if len(word) > 6:
            category_match.append('greater_than_6')
        return category_match


def reappStrategyFactory(strategyString):
    if strategyString == farAwayFlag:
        return SpatioTempStrategy()
    elif strategyString == objectiveFlag:
        return ObjectiveStrategy()
    else:
        raise Exception("Incorrect Strategy")


@dataclass
class SentimentWrapper:
    polarity: float
    objectivity: float