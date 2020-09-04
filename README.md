# Linguistic Distancing Model

## __Dependencies__

Dependencies are tracked using `pipenv` within `Pipfile. Instructions for installation of ```pipenv``` are located [here](https://pipenv.kennethreitz.org/en/latest/install/#installing-pipenv).

To start up the virtual environment and install all dependencies:

```shell
pipenv --python [version]
```
The version of Python installed should be at least `3.6`.

## __Running__

To run the algorithm, call the following in the command line while in the repository's main directory:

```shell
python model -args
```

```
usage: model [-h] [-e EVAL] [-f] [-o] [-s SAVEMODEL] [-l LOAD_MODEL] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -e EVAL, --eval EVAL  Specify a path of a csv or excel to be evaluated. Add
                        if you're not testing for correlation of testing data
                        and just want results.
  -f, --farAway         Reappraise using far-away distancing
  -o, --objective       Reappraise using objective distancing
  -s SAVEMODEL, --saveModel SAVEMODEL
                        Specify a location to save the model once it has been
                        trained
  -l LOAD_MODEL, --load-model LOAD_MODEL
                        Specify a previously trained model to be loaded.
  -d, --dev             Used to print debug statement
```

At least one reappraisal strategy (either far-away or objective) must be specified in order to run the algorithm. Specifying both will test both reappraisal strategies.

## __Algorithm__

By default, the algorithm operates on two files:

- `eval/data_train_fixed.csv`
- `eval/data_test_fixed.csv`

The algorithm trains the model based on the values in the first file and tests based on the values in the  second. See "Data Format" to see the formatting of the training data.

In order to train the data, the user must specify which columns to use as the responses, and which is used as the score.

After training, the user will be prompted via file dialog to select a table of test data.

### __Model Fitting__

To fit the model, the model first predicts the part-of-speech tag for each word in each response using SpaCy. Then, two things happen:

#### Score Extrapolation

For each response, each word-tag pair is classified into specific predetermined categories, and the model counts the number of positive category matches and the number of negative category matches. From these counts and the raw score of the response, the model extrapolates a positive score and a negative score, which serve as a base score. The extrapolation of this score satisfies two requirements:

- The sum of all scores, positive and negative, equals the raw response score.
- All positive category matches contribute an equal amount of positive score, and all negative category matches contribute an equal amount of negative score.

The model then applies those scores to each word-tag pair and also applies the score to each category.

#### Weight Extrapolation

The number of categories is counted and placed into a count vector. Each vector becomes a row in a matrix of weights. Given the score of the response, the model performs a least-squares best fit to find the optimal weighting of each category.

After applying this process to every response, the model then transforms the word-tag pair into a tag->word->score mapping, where the score is an average of all occurences of word-tag scores.

### __Model Prediction__

To predict the score of text, the model tags the input text and then simply looks up the score of each word based on the trained tag-word dictionary. If the word is a novel word (thus not in the dictionary), the model searches for synonyms of that word within the mapping and finds the average of all the synonym's scores. Once this is found, the dictionary is updated with that word-tag's score.
