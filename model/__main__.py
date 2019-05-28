import reappraisal

import pandas as pd
import spacy

import os
import sys
import threading
import logging

FORMAT = '%(asctime)-15s: %(message)s'


def main():
    strat = None
    verbose = False
    for arg in sys.argv[1:]:
        if arg == "-v":
            verbose = True
            if verbose:
                logging.basicConfig(level=logging.DEBUG, format=FORMAT)
                logging.debug("Verbose Logging Enabled")
            else:
                logging.basicConfig(level=logging.INFO, format=FORMAT)
        elif arg == "-f":
            strat = "f"
            logging.info("Far Away Analysis Initialized")
        elif arg == "-o":
            strat = "o"
            logging.info("Objective Analysis Initialized")

        else:
            raise(Exception("Invalid Argument: {0}".format(arg)))

    cwd = os.getcwd()
    data = pd.DataFrame(columns = ['Text Response', "Objectivity Score", "Far Away Score"])
    for filename in os.listdir(cwd + "/input/training"):
        data = pd.concat([data, reappraisal.extrapolate_data(cwd + "/input/training/" + filename)], axis = 0)
    if not strat:
        strat = input("Press f to initiate a far-away model. Press o to initiate an objectivity model.  ")
    model = reappraisal.Model(data, strat, verbose)
    model.fit()

    # Reading Test Data
    test_data = pd.DataFrame(columns = ['Text Response', "Objectivity Score", "Far Away Score"])
    for filename in os.listdir(cwd + "/input/test"):
        test_data = pd.concat([test_data, reappraisal.extrapolate_data(cwd + "/input/test/" + filename)], axis = 0)
    logging.info(f"{len(test_data)} responses added for testing.")
    if strat == 'f':
        ## Far Away Labeling
        test_data.columns = ["Text Response", "Observed Score", "Expected Score"]
    elif strat == 'o':
        ## Objective Labeling 
        test_data.columns = ["Text Response", "Expected Score", "Observed Score"]
    test_data['Observed Score'].values[:] = float('-inf')

    ## Predict Scores from Test Data 
    for index, row in test_data.iterrows():
        response = row['Text Response']
        scored_sentence, score = model.predict(response)
        test_data.iloc[index, test_data.columns.get_loc('Observed Score')] = score

    test_data.to_excel(f"{cwd}/output/{strat}_test.xlsx")



main()
