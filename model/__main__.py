import reappraisal

import pandas as pd
import spacy

import os
import sys
import threading

def main():
    strat = None
    verbose = False
    for arg in sys.argv[1:]:
        if arg == "-v":
            verbose = True
        elif arg == "-f":
            strat = "f"
        elif arg == "-o":
            strat = "o"
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
    # model.predict("I really love her.")

    test_data = pd.DataFrame(columns = ['Text Response', "Objectivity Score", "Far Away Score"])
    for filename in os.listdir(cwd + "/input/test"):
        test_data = pd.concat([test_data, reappraisal.extrapolate_data(cwd + "/input/training/" + filename)], axis = 0)
    # TODO: for row of test data, model.predict(test_text_response)
    #rando change





main()
