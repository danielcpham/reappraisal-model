import data_process

import numpy as np
import pandas as pd

from collections import defaultdict
import logging

import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc, Token
# from spacy_wordnet.wordnet_annotator import WordnetAnnotator 

##TODO: sentiment after scoring entire sentence instead of scoring entire word?? 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()

FORMAT = '%(asctime)-15s: %(message)s'


class Model:
    def __init__(self, df, strat = 'o', verbose = False):
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format=FORMAT)
            logging.debug("Verbose Logging Enabled")
        else:
            logging.basicConfig(level=logging.INFO, format=FORMAT)
        logging.info("Model Created")
        
        ### Class variable initialization
        self.wordtag_scores = {}
        self.weights = {}
        self.strat = strat
        self.nlp = spacy.load('en_core_web_md')
        logging.debug("spaCy Library Loaded")

    
        #TODO: replace wordnet with spacy synonyms
        # self.nlp.add_pipe(WordnetAnnotator(self.nlp.lang), 'synsets', after='tagger')

        if strat == 'f':
            self.df = df[['Text Response', 'Far Away Score']]
            self.reappStrategy = data_process.reappStrategyFactory('spatiotemp')
        elif strat == 'o':
            self.df = df[['Text Response', 'Objectivity Score']]
            self.reappStrategy = data_process.reappStrategyFactory('obj')
            ### Initialize sentiment analysis 
        else:
            raise Exception("Please use either 'f' for Spatial Analysis or 'o' for Objective Analysis")
        self.df.columns = ['Response', 'Score']
        # for word in STOP_WORDS:
        #     ### Initialize stop words.
        #     for w in (word, word[0].capitalize(), word.upper()):
        #         lex = self.nlp.vocab[w]
        #         lex.is_stop = True
        self.data = []
        logging.debug("Model Initialized")


    def fit(self):
        ### Iterate through all the rows in the data 
        for _ , row in self.df.iterrows():
            response, score = row['Response'].lower(), row['Score']
            ### Process using SpaCy
            doc = self.nlp(response)
            tagged_response = []
            ### Add the tagged score to the data, ignoring stop words and punctuation
            for token in doc:
                if not token.is_punct: 
                    tagged_response.append((token.tag_,token.lemma_, 0))
            ### Add the tagged response to the dataset
            self.data.append((tagged_response, score))
        logging.debug("Training Data Preprocessed")
        full_score_list = []
        weights_list = []
        ### For each response in the data set, train the model on the expected scores
        for response, score in self.data:
            weights, score_list = self.fit_word_scores(response, score)
            full_score_list.append(score_list)
            weights_list.append((weights, score))

        word_scores = self.get_scoring_bank(full_score_list)
        observed_weights = self.best_fit_weights(weights_list)
        
        self.weights = observed_weights
        self.wordtag_scores = word_scores
        logging.info(f"Model trained on {len(self.df)} responses")

        
    def predict(self, text):
        doc = self.nlp(text)
        scored_sentence = []
        for token in doc:
            print(token)
            score = 0
            if not  token.is_punct:
                category_match = self.reappStrategy.classifier(token.lemma_, token.tag_)
                if token.tag_ in self.wordtag_scores:
                    # logging.debug(f"Tag {token.tag_} exists in scoring bank")
                    if token.lemma_ in self.wordtag_scores[token.tag_]:
                        ### Word found in bank
                        logging.debug(f"WordTag ({token.lemma_},{token.tag_}) exists in scoring bank")
                        score = self.wordtag_scores[token.tag_][token.lemma_]
                        if category_match:
                            ### TODO: Category score multiplier
                            logging.debug(f"Categories {category_match}")
                            score = score
                    else:
                        ### TODO: How to deal with words that aren't in the training data
                        ### TODO: Word sense distinguisher for synonyms 
                        ### TODO: look for semantically similar words 
                        score = 0
                    if self.strat == "o":
                        ### TODO: sentiment analysis
                        score = score
                        # print(token.lemma_, token._.polarity)
                print(token.lemma_, token.tag_, score)

            ### Add the token and the score to the scored list. 
            scored_sentence.append((token.text, score))
        return scored_sentence

    def standardize_weights(self, weights): 
        """ 
        Standardizes the weights vector.

        Arguments:
            weights {dict} -- vector of weight scores for each category
        
        Returns:
            {dict} -- weights modified to include all weights
        """
        for cat in self.reappStrategy.categories:
            if cat not in weights:
                weights[cat] = 0
        return weights

    def get_scoring_bank(self, scored_list):
            """
            :param scored_list: a list of scored words of the form ((Tag, Word), Score)
            :return: the list aggregated as a dictionary mapping Tag ->Word -> Score
            """
            score_dict = defaultdict(dict)
            for response in scored_list:
                for word, tag, score in response:
                    if word not in score_dict[tag]:
                        score_dict[tag].update({word: []})
                    score_dict[tag][word].append(score)
            for tag in score_dict:
                for word in score_dict[tag]:
                    score_dict[tag][word] = np.mean(score_dict[tag][word])
            score_dict = dict(score_dict)
            # print("SCORE DICTIONARY: {0}".format(score_dict))
            return score_dict

    def fit_word_scores(self, tagged_response, score):
        """
        :param taggedResponse: the tagged response
        :param score: the score of the tagged response
        :param reappStrategy: the strategy of reappraisal
        :return: dictionary of weights for each category, list of scores for each word
        """
        pos_list = []
        neg_list = []
        score_list = []
        weights = defaultdict(int)
        ### Score each word-tag pair based on the categories it fits in
        for index in range(len(tagged_response)):
            tag = tagged_response[index][0]
            word = tagged_response[index][1]
            ### Classify the word-tag pair based on the strategy used
            matched_categories = self.reappStrategy.classifier(word, tag)
            ### Separate positive and negative category matches 
            for category in matched_categories:
                if category in self.reappStrategy.posCategories:
                    pos_list.append((word, tag))
                    weights[category] += 1
                if category in self.reappStrategy.negCategories:
                    neg_list.append((word, tag))
                    weights[category] += 1
        ### Determine the correct positive/negative score based on the data. 
        ### Only negative categories exist
        if len(pos_list) == 0:
            pos_score = 0
            neg_score = 0 if len(neg_list) == 0 else score / len(neg_list)
        ## Only positive categories exist 
        elif len(neg_list) == 0:
            pos_score = score / len(pos_list)
            neg_score = 0
        ### Both lists have categories appear in the sentence
        else:
            if len(pos_list) == len(neg_list):
                neg_score = -score / abs(len(pos_list) + len(neg_list))
                pos_score = -neg_score / (score / len(pos_list))
            else:
                pos_score = score / abs(len(pos_list) - len(neg_list))
                neg_score = -pos_score
        ### Obtain the word,tag,score tuple for each positive word
        for word, tag in pos_list:
            score_list.append((word, tag, pos_score))
        ### Obtain the word,tag,score tuple for each negative word
        for word, tag in neg_list:
            score_list.append((word, tag, neg_score))           
        ### For each category in the weights matrix, get the raw score:
        ###     Number of occurrences of the weight * score = raw score 
        for category in weights:
            if category in self.reappStrategy.posCategories:
                weights[category] *= pos_score
            if category in self.reappStrategy.negCategories:
                weights[category] *= abs(neg_score)
        ### Standardize the weights matrix to include all categories
        standard_weights = self.standardize_weights(dict(weights))
        return standard_weights, score_list
           
    def best_fit_weights(self, weights_list):
        """ Calculates the best fit of the weights and the expected score
        Arguments:
            weights_list -- list of tuples containing:
                - weights_dict -- dictionary containing the weighted scores 
                    of each category of a response
                - expected_score -- the score of that response
        Returns:
            A single dictionary reprsenting the weights that best fit the data
        """
        weight_matrix = pd.DataFrame(columns = self.reappStrategy.categories)
        expected_score = []
        ### Convert each weight to a vector
        for weights_dict, score in weights_list:
            weights_row = pd.DataFrame([weights_dict], dtype = 'float')
            ### Add the next row at the bottom of the weights matrix
            weight_matrix = pd.concat((weight_matrix, weights_row), ignore_index = True, sort = False)
            expected_score.append(score)
        ### Calculate the least squares best fit 
        observed_weights = np.linalg.lstsq(weight_matrix.values, np.array(expected_score, dtype = 'float'), rcond = None)[0]
        return pd.DataFrame(observed_weights, index = self.reappStrategy.categories).to_dict()[0]


    
def extrapolate_data(filename):
    """
    Arguments:
        filename {string} --  the absolute path for an .xlsx file 
    Returns:
        Pandas dataframe -- the xlsx or csv file converted into a Pandas dataframe
    """
    if 'xlsx' in filename:
        df = pd.read_excel(filename)
        df = df.iloc[:, 0:3].dropna(thresh=3)
        df.columns = ['Text Response', "Objectivity Score", "Far Away Score"]
    else:
        df = pd.read_csv(filename)
        df.columns = ['Text Response', "Objectivity Score", "Far Away Score"]
    return df

def polarity(token):
    sentiment_analyzer = sentiment
    if not token.is_punct:
        return sentiment.polarity_scores(token.text) 


    
#TODO: add Synonym 
#Lesk in Spacy https://sp1819.github.io/wordnet_spacy.pdf
class Synonyms:
    def __init__(self):
        return 

#TODO: add the reappraisal as a new pipeline


               



        