from data_process import reappStrategyFactory, SentimentWrapper


import numpy as np
import pandas as pd

from collections import defaultdict
import logging
from dataclasses import dataclass

import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc, Token
from textblob import TextBlob

from nltk.wsd import lesk
from nltk.sentiment import vader

FORMAT = '%(asctime)-15s: %(message)s'


class Model:
    def __init__(self, df: pd.DataFrame, strat = 'o', verbose = False):
        if verbose:
            logging.basicConfig(level=logging.DEBUG, format=FORMAT)
        else:
            logging.basicConfig(level=logging.INFO, format=FORMAT)
        ### Class variable initialization
        self.wordtag_scores = {}
        self.weights = {}
        self.strat = strat
        self.nlp = spacy.load('en_core_web_md')
        logging.debug("spaCy Library Loaded")

        if strat == 'f':
            self.df = df[['Text Response', 'Far Away Score']]
            self.reappStrategy = reappStrategyFactory('spatiotemp')
        elif strat == 'o':
            self.df = df[['Text Response', 'Objectivity Score']]
            self.reappStrategy = reappStrategyFactory('obj')

            ### Initialize sentiment analysis 
            Doc.set_extension("sentiment",
            getter = lambda doc: TextBlob(doc.text).sentiment)
        else:
            raise Exception("Please use either 'f' for Spatial Analysis or 'o' for Objective Analysis")
        self.df.columns = ['Response', 'Score']
        self.data = []
        logging.debug("Model Initialized")


    def fit(self):
        '''
        Fits all of the data given according to the weights of the given strategy's categories. 
        '''
        full_score_list = []
        weights_list = []
        ### Iterate through all the rows in the data 
        for _ , row in self.df.iterrows():
            response, score = row['Response'].lower(), row['Score']
            ### Process using SpaCy
            doc = self.nlp(response)
            tagged_response = []
            ### Add the tagged score to the data, ignoring stop words and punctuation
            for token in doc:
                if not token.is_punct: 
                    word = token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower()
                    tag = token.tag_
                    tagged_response.append((tag, word))
        
            # PROCESS TAGGED RESPONSE 
            # If objective, first multiply the score by the sentiment score
            #   and then fit the words to that modified score.
            
            if self.strat == "o":
                # Normalize the sentiment. 
                sentiment = SentimentWrapper(doc._.sentiment.polarity, doc._.sentiment.subjectivity)
                sentiment = normalize_sentiment(sentiment)
                # Let the sentiment modifier be the average of polarity and subjectivity scores. 
                score *= (sentiment.polarity + sentiment.subjectivity) / 2
                logging.debug(f"New Score: {score}, Polarity: {sentiment.polarity}. Subjectivity: {sentiment.subjectivity}")

            weights, score_list = self.fit_word_scores(tagged_response, score)
            full_score_list.append(score_list)
            weights_list.append((weights, score))

        #     if self.strat == "o":

        #         sentiment = doc._.sentiment
        #         if sentiment.polarity == 0:
        #             sentiment.polarity += 0.001
        #         sentiment.polarity /= 5

        #         if sentiment.subjectivity == 0:
        #             sentiment.subjectivity += 0.001
        #         sentiment.subjectivity /= 5
            
        #     self.data.append((tagged_response, score))
        
        # ### For each response in the data set, train the model on the expected scores
        # for response, score in self.data:
        #     weights, score_list = self.fit_word_scores(response, score)
        #     full_score_list.append(score_list)
        #     weights_list.append((weights, score))
        word_scores = self.get_scoring_bank(full_score_list)
        observed_weights = self.best_fit_weights(weights_list)
        self.weights = observed_weights
        self.wordtag_scores = word_scores
        logging.info(f"Model trained on {len(self.df)} responses")



    def predict(self, text: str):
        scored_sentence = []
        doc = self.nlp(text)
        for token in doc:
            word = token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower()
            tag = token.tag_
            score = 0
            if not token.is_punct:
                logging.debug(f"({word},{tag})")
                category_match = self.reappStrategy.classifier(word, tag)
                if tag in self.wordtag_scores:
                    if word in self.wordtag_scores[tag]:
                        # Word found in bank
                        logging.debug(f"({word},{tag}) exists in scoring bank")
                        score = self.wordtag_scores[tag][word]
                        if category_match:
                            # Multiplies the raw score by the weight of the category  
                            logging.debug(f"({word},{tag}) Categories: {category_match}")
                            for category in category_match:
                                score *= self.weights[category]
                    else:
                        # Word not found in bank; search synonyms 
                        synonyms = get_synonyms(text, word, tag)
                        sim_score = 0
                        count = 1
                        if synonyms:
                            logging.debug(f"Synonyms of {word}: {synonyms}")
                            for synonym in synonyms:
                                if synonym in self.wordtag_scores[tag]:
                                    # Synonym found in the bank 
                                    count += 1
                                    logging.debug(f"Synonym For {word}: {synonym} -> {self.wordtag_scores[tag][synonym]}")
                                    sim_score += self.wordtag_scores[tag][synonym]
                            sim_score /= count # Get the average of matching synonym scores 
                            # Save the result in the bank
                            self.wordtag_scores[tag][word] = sim_score
                        score = sim_score
            ### Add the token and the score to the scored list. 
            scored_sentence.append((token.text, score))
        logging.debug(scored_sentence)
        
        total_score = sum([score for word, score in scored_sentence])
        # Objectivity check: "undo" the multiplication by sentiment scores. 
        if self.strat == "o":
            sentiment = SentimentWrapper(doc._.sentiment.polarity, doc._.sentiment.subjectivity)
            sentiment = normalize_sentiment(sentiment)
            total_score /= (sentiment.polarity + sentiment.subjectivity) / 2
        return scored_sentence, total_score

    def standardize_weights(self, weights: dict): 
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

    def fit_word_scores(self, tagged_response: list, score: float):
        """
        :param taggedResponse: a single tagged response
        :param score: the score of the tagged response
        :return: dictionary of weights for each category, list of scores for each word
        """
        pos_list = []
        neg_list = []
        score_list = []
        weights = defaultdict(int)
        ### Score each word-tag pair based on the categories it fits in
        for tag, word in tagged_response:
            ### Classify the word-tag pair based on the strategy used
            matched_categories = self.reappStrategy.classifier(word, tag)
            # synonyms = get_synonyms(list(map(lambda wordTag: wordTag[0], tagged_response)), word, tag)
            ### Separate positive and negative category matches 
            if len(matched_categories) != 0:
                for category in matched_categories:
                    if category in self.reappStrategy.posCategories:
                        pos_list.append((word, tag))
                        weights[category] += 1
                    if category in self.reappStrategy.negCategories:
                        neg_list.append((word, tag))
                        weights[category] += 1
            ### Determine the correct positive/negative score based on the data. 
        ### Only negative categories exist
        # logging.debug(tagged_response)
        # logging.debug(f"Score, LenPosList, LenNegList: ({score},{len(pos_list)},{len(neg_list)})")
        if len(pos_list) == 0:
            pos_score = 0
            neg_score = 0 if len(neg_list) == 0 else score / len(neg_list)
        ## Only positive categories exist 
        elif len(neg_list) == 0:
            pos_score = 0
            neg_score = 0 if len(pos_list) == 0 else score / len(pos_list)
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
        # logging.debug(score_list)
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
        filename {string} --  the absolute path for an .xlsx or .csv file 
    Returns:
        Pandas dataframe -- the xlsx or csv file converted into a Pandas dataframe
    """
    if 'xlsx' in filename:
        df = pd.read_excel(filename)
        df = df.iloc[:, 0:3].dropna(thresh=3)
        df.columns = ['Text Response', "Objectivity Score", "Far Away Score"]
    else:
        #TODO: check if valid file extension. 
        df = pd.read_csv(filename)
        df.columns = ['Text Response', "Objectivity Score", "Far Away Score"]
    return df


# def most_similar(token):
#     """Returns the 10 most similar words of the token. 
#     Arguments:
#         token {Token} -- Token to be searching for synonyms 
#     Returns:
#         list(str)-- string of synonyms for token
#     """
#     queries = [t for t in token.vocab if t.is_lower == token.is_lower and t.prob >= -15]
#     by_similarity = sorted(queries, key=lambda t: token.similarity(t), reverse=True)
#     return by_similarity[:np.minimum(10, len(by_similarity))]  

def convert_to_wordnet(tag):
    """
    :param tag: POS tag as defined by Penn Treebank
    :return: POS tag for use in wordnet
    """
    if tag in {'NN', 'NNS', "NNP", 'NNPS', "n"}:
        return 'n'
    elif tag in {'VB', 'VBD', 'VBP', 'VBZ', 'v'}:
        return 'v'
    elif tag in {'a','JJ', 'JJR', 'JJS'}:
        return 'a'
    elif tag in {'r', 'RB', 'RBR', 'RBS', 'WRB'}:
        return 'r'
    else:
        return None

def get_synonyms(sentence, word, tag=None):
    """
    :param: sentence: sentence with which to grab context, respresented as a list of words
    :param word: a word to synonymize
    :param tag: optional, specifies POS
    :return: list of synonyms of that word

    Uses the lesk WSD algorithm to obtain the most likely set of synonyms. 
    """
    if word not in STOP_WORDS:
        wn_tag = convert_to_wordnet(tag) if tag else None
        if wn_tag:
            probable_synset = lesk(sentence, word, pos=wn_tag)
        else:
            probable_synset = lesk(sentence, word)
        if probable_synset:
            synonyms = set(
                filter(lambda lemma: (not "_" in lemma.name()) and (not lemma.name() == word), probable_synset.lemmas()))
            return list(map(lambda lemma: lemma.name().lower(), synonyms))
        else:
            return []
    return []


def normalize_sentiment(sentiment):
    '''
    Converts sentiment:

    Polarity ranges from [-1, 1] (By TextBlob API). If the polarity has absolute value 1 (-1 or 1), then set it to 0.01 to 
        avoid getting a 0 value. Else, take the negative and add 1 to get the score between [0, 1]. 

    Sentiment ranges from [0, 1] (By Textblob API). If sentiment has absolute value 0, set it 0.01 to avoid getting 
        a 0 value. Else, leave it. 
    '''

    sentiment.polarity = 0.01 if np.abs(sentiment.polarity) == 1 else -np.abs(sentiment.polarity) + 1 
    if sentiment.subjectivity == 0:
        sentiment.subjectivity += 0.01
    # if sentiment.polarity == 0:
    #     # Must take absolute value because polarity ranges from -1 to 1 (TextBlob API)
    #     sentiment.polarity += 0.1
    #     sentiment.polarity /= 5
    return sentiment          



        