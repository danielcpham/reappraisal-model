from data_process import reappStrategyFactory, SentimentWrapper


import numpy as np
import pandas as pd

from collections import defaultdict
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import sys

import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokens import Doc, Token
from textblob import TextBlob

from nltk.wsd import lesk
from nltk.sentiment import vader

FORMAT = '%(asctime)-15s: %(message)s'


class Model:
    def __init__(self, df: pd.DataFrame, starttime, strat = 'o', verbose = False):
        # Initialization for logging
        self.logger = logging.getLogger("REAPPRAISAL")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(name)s-%(levelname)s]: %(asctime)-15s: %(message)s')

        starttime = datetime.now().strftime('%y%m%d_%H%M%S')
        fh = logging.FileHandler(f'output/{starttime}-{strat}.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)

        if verbose: 
            fhv = logging.FileHandler(f'output/verbose/{starttime}-{strat}.log')
            fhv.setLevel(logging.DEBUG)
            fhv.setFormatter(formatter)
            self.logger.addHandler(fhv)

        ### Class variable initialization
        self.wordtag_scores = {}
        self.weights = {}
        self.strat = strat

        # Attempts to load language model en_core_web_md. If it doesn't load, 
        # it runs a script that downloads it from the internet.
        try: 
            nlp = spacy.load('en_core_web_md')
        except OSError:
            os.system('python -m spacy download en_core_web_md')
            nlp = spacy.load('en_core_web_md')
        self.nlp = nlp
        
        self.logger.info("spaCy Library Loaded")

        # Checks the reappraisal strategy, filters data appropriately,
        # and initializes the reappraisal strategy
        if strat == 'f':
            self.df = df[['Text Response', 'Far Away Score']]
            self.reappStrategy = reappStrategyFactory('spatiotemp')
        elif strat == 'o':
            self.df = df[['Text Response', 'Objectivity Score']]
            self.reappStrategy = reappStrategyFactory('obj')
            # Initialize sentiment analysis at the sentence level/
            Doc.set_extension("sentiment",
            getter = lambda doc: TextBlob(doc.text).sentiment)
        else:
            raise Exception("Please use either 'f' for Spatial Analysis or 'o' for Objective Analysis")
        self.df.columns = ['Response', 'Score']
        self.data = []
        self.logger.debug("Model Initialized")


    def fit(self):
        """
        Fits all of the data given according to the weights of the given strategy's categories. 
        """
        full_score_list = []
        weights_list = []
        # Iterate through all the rows in the data 
        for _ , row in self.df.iterrows():
            response, score = row['Response'].lower(), row['Score']
            # Creates a Doc object based on the single response 
            doc = self.nlp(response)
            tagged_response = []
            # For each token in the document, add the tagged word to the data, 
            # ignoring stop words and punctuation
            for token in doc:
                if not token.is_punct: 
                    word = token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower()
                    tag = token.tag_
                    tagged_response.append((tag, word))
            # PROCESS TAGGED RESPONSE 
            # If objective, first multiply the score by the sentiment score
            #   and then fit the words to that modified score.

            # If analyzing objective response, gets the sentiment scores of the sentence 
            # and adjust tagged score proportionally. 
            # See normalize_sentiment() for normalization procedures. 
            if self.strat == "o":
        
                sentiment = SentimentWrapper(doc._.sentiment.polarity, doc._.sentiment.subjectivity)
                sentiment = normalize_sentiment(sentiment)
                sentiment_proportion = (sentiment.subjectivity)
                self.logger.debug(f'Old Score: {score}, New Score: {score * sentiment_proportion}')
                score *= sentiment_proportion
            # After sentence sentiment is taken into account, fit the scores at the word level.
            # Returns the weights dictionary of the principal components and a list of word-score tuples. 
            weights, score_list = self.fit_word_scores(tagged_response, score)
            full_score_list.append(score_list)
            weights_list.append((weights, score))
        # Collapses each word score list into a dictionary.
        # Calculates least squares best fit of weights.
        word_scores = self.get_scoring_bank(full_score_list)
        observed_weights = self.best_fit_weights(weights_list)
        self.weights = observed_weights
        self.wordtag_scores = word_scores
        self.logger.info(f"Model trained on {len(self.df)} responses")

    def predict(self, text: str):
        """
        Predicts reappraisal of text. 
        
        Arguments:
            text {str} -- The text to be analyzed.
        
        Returns:
            Scored response, a list of word score pairs
            Total score of the response
        """
        scored_sentence = []
        doc = self.nlp(text)
        for token in doc:
            word = token.lemma_ if token.lemma_ != "-PRON-" else token.text.lower()
            tag = token.tag_
            score = 0
            if not token.is_punct:
                self.logger.debug(f"({word},{tag})")
               
                if tag in self.wordtag_scores:
                    if word in self.wordtag_scores[tag]:
                        # Word-tag pair found in scoring bank. 
                        self.logger.debug(f"({word},{tag}) exists in scoring bank")
                        score = self.wordtag_scores[tag][word]
                        # Gets the categories that the word-tag pair falls into.
                        # If any categories match,
                        # multiplies the raw score by the weight of the category  
                        category_match = self.reappStrategy.classifier(word, tag)
                        if category_match:
                            self.logger.debug(f"({word},{tag}) Categories: {category_match}")
                            for category in category_match:
                                score *= self.weights[category]
                    else:
                        # Word-tag pair not found in bank; search synonyms 
                        synonyms = get_synonyms(text, word, tag)
                        sim_score = 0
                        count = 1
                        if synonyms:
                            self.logger.debug(f"Synonyms of {word}: {synonyms}")
                            for synonym in synonyms:
                                if synonym in self.wordtag_scores[tag]:
                                    # Synonym found in the bank 
                                    count += 1
                                    self.logger.debug(f"Synonym For {word}: {synonym} -> {self.wordtag_scores[tag][synonym]}")
                                    sim_score += self.wordtag_scores[tag][synonym]
                            sim_score /= count # Get the average of matching synonym scores 
                            # Save the result in the bank
                            self.wordtag_scores[tag][word] = sim_score
                        score = sim_score
            ### Add the token and the predicted score to the scored list. 
            scored_sentence.append((token.text, score))
        self.logger.debug(scored_sentence)
        # Sums up the scores of the entire text. 
        total_score = sum([score for word, score in scored_sentence])
        # If analyzing objective distancing, "undo" the proportion of sentiment in the score. 
        if self.strat == "o":
            sentiment = SentimentWrapper(doc._.sentiment.polarity, doc._.sentiment.subjectivity)
            sentiment = normalize_sentiment(sentiment)
            sentiment_proportion = sentiment.subjectivity
            total_score /= sentiment_proportion
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
        # Score each word-tag pair based on the categories it fits in
        for tag, word in tagged_response:
            # Classify the word-tag pair based on the strategy used
            matched_categories = self.reappStrategy.classifier(word, tag)
            # Separate positive and negative category matches into separate lists 
            if len(matched_categories) != 0:
                for category in matched_categories:
                    if category in self.reappStrategy.posCategories:
                        pos_list.append((word, tag))
                        weights[category] += 1
                    if category in self.reappStrategy.negCategories:
                        neg_list.append((word, tag))
                        weights[category] += 1
        # Determine the correct positive/negative score based on the data. 
        if len(pos_list) == 0:
            # Case where only negative categories exist
            pos_score = 0
            neg_score = 0 if len(neg_list) == 0 else score / len(neg_list)
        elif len(neg_list) == 0:
            # Case where only positive categories exist 
            neg_score = 0
            pos_score = 0 if len(pos_list) == 0 else score / len(pos_list)
        else:
            # Case where both positive and negative categories exist 
            if len(pos_list) == len(neg_list):
                # Special subcase where there's an equal number of positive and negative categories
                neg_score = -score / abs(len(pos_list) + len(neg_list))
                pos_score = -neg_score / (score / len(pos_list))
            else:
                pos_score = score / abs(len(pos_list) - len(neg_list))
                neg_score = -pos_score
        # Obtain the word,tag,score tuple for each positive word
        for word, tag in pos_list:
            # Check sentiment at word level when analyzing objective distancing. 
            if self.strat == 'o':
                self.logger.debug(f'Original: {word} -> {pos_score}')
                textblob = TextBlob(word).sentiment
                sentiment = SentimentWrapper(textblob.polarity, textblob.subjectivity)
                sentiment = normalize_sentiment(sentiment)
                sentiment_score = (sentiment.subjectivity + sentiment.polarity) / 2 
                self.logger.debug(f'After Sentiment: {word} -> {pos_score * sentiment_score}')
                pos_score *= sentiment_score
            score_list.append((word, tag, pos_score))
        # Obtain the word,tag,score tuple for each negative word
        for word, tag in neg_list:
        
            # Check sentiment at word level when analyzing objective distancing. 
            # if self.strat == 'o':
            #     self.logger.debug(f'Original: {word} -> {neg_score}')
            #     textblob = TextBlob(word).sentiment
            #     sentiment = SentimentWrapper(textblob.polarity, textblob.subjectivity)
            #     sentiment = normalize_sentiment(sentiment)
            #     sentiment_score = (sentiment.subjectivity + sentiment.polarity) / 2 
            #     self.logger.debug(f'After Sentiment: {word} -> {neg_score * sentiment_score}')
            #     neg_score *= sentiment_score
            score_list.append((word, tag, neg_score))    
        # For each category in the weights matrix, get the raw score:
        #   raw_score = Number of occurrences of the weight * score 
        for category in weights:
            if category in self.reappStrategy.posCategories:
                weights[category] *= pos_score
            if category in self.reappStrategy.negCategories:
                weights[category] *= abs(neg_score)
        # Standardize the weights matrix to include all categories
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
        # Calculate the least squares best fit 
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

######################
## HELPER FUNCTIONS ##
######################

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
    """
        Converts sentiment:

        Polarity ranges from [-1, 1] (By TextBlob API). If the polarity has absolute value 1 (-1 or 1), 
            then set it to 0.01 to avoid getting a 0 value. 
            Else, take the negative and add 1 to get the score between [0, 1]. 

        Subjectivity ranges from [0, 1] (By Textblob API). If subjectivity has absolute value 0,
            then set it 0.01 to avoid getting a 0 value. 
            Else, leave it. 

        High subjectivity/low polarity approaches 1
        Low subjectivity/high polarity approaches 0
    """
    sentiment.polarity = 0.01 if np.abs(sentiment.polarity) == 1 else -np.abs(sentiment.polarity) + 1 
    sentiment.subjectivity = 0.01 if sentiment.subjectivity == 0 else sentiment.subjectivity
    return sentiment          



        