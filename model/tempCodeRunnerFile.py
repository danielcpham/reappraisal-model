import numpy as np
import pandas as pd
import os

# cwd = os.getcwd()

# def extrapolate_data(filename):
#     """
#     Arguments:
#         filename {string} --  the absolute path for an .xlsx file 
#     Returns:
#         Pandas dataframe -- the xlsx or csv file converted into a Pandas dataframe
#     """
#     if 'xlsx' in filename:
#         df = pd.read_excel(filename)
#         df = df.iloc[:, 0:3].dropna(thresh=3)
#         df.columns = ['Text Response', "Objectivity Score", "Far Away Score"]
#     else:
#         df = pd.read_csv(filename)
#         df.columns = ['Text Response', "Objectivity Score", "Far Away Score"]
#     return df

# for filename in os.listdir(cwd + "\input"):
#     data = extrapolate_data(cwd + "\input\\" + filename)
# print(data)



# def verboseprint(*args, **kwargs): 
#     args = list(args)
#     if True in args:
#         args.pop(True)
#         print(*args, **kwargs)
#     # if "verbose" in kwargs:
#     #     if kwargs['verbose']:
#     #         kwargs.pop('verbose')
#     #         print(*args, **kwargs)
#     # for kw in kwargs.items():
#     #     print(kw)
#     # print(*args, **kwargs)

# verboseprint("Hello", True)

import spacy 
try:
    spacy.load("de_core_news_sm")
except OSError:
    print('hello')
