# Linguistic Distancing Model 

## __Dependencies__ :
- Python 3.x+
- spaCy 2.x+
    - ```python -m spacy download en_core_web_md```
- NLTK 3.x+ (for now)
- TextBlob 
- numPy 1.16+
- Pandas 0.23.4+
- Future Dependencies:
    - LIWC 

All current dependencies (given that Python is installed on the device) can be installed with the following:
```
pip install xxx
```

To install the models needed to run the algorithm, run the following in command:

```
python -m spacy download en_core_web_sm
```

## __Running__: 

To run the algorithm, call the following in the command line while in the repository's main directory:

```shell
python model -args
```
### Additional Arguments:
- ```-f ``` - Far Away Analysis - Flag to enable analysis for far away/spatiotemporal distancing. 
- ```-o ``` - Objectivity Analysis - Flag to enable analysis for objective distancing. 
    - Note that the ``` -f/-o``` flags are mutually exclusive; the script is currently incapable of analyzing far away distancing and objective distancing simultaneously. 
- ```-v ``` : verbose - Flag to enable debug logging. 



## __Algorithm__:
- The ```input\training``` directory includes the data from Experimetrix and LDH Phase 1 that was used to train the model. See "Data Format" to see the formatting of the training data.
- When the model is called, the model will automatically pull any database files from ```input\training``` and begins to fit the model based on that data. 
- If a strategy flag was included in the command line, then it will automatically learn the data on that reappraisal strategy. Otherwise, the user will be prompted to enter one. 
- After training, the user will be prompted via file dialog to select a table of test data. The test data should be formatted similarly to the formatting shown below. 

## __Data Format__:

| Response | Objectivity Score | Far Away Score | 
| -------- | ----------------- | -------------- |
| A utf-8 string of the sentence being analyzed. | The average score, represented as a float, given by human coders for the corresponding sentence for objectivity. | The average score, represented as a float, given by human coders for the corresponding sentence for far away distancing. | 



## __Notes__:



## __Future Updates__:
- Set logging level with -v and a number 
- Ability to specify training data


