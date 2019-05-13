# Linguistic Distancing Model 

## __Dependencies__ :
- Python 3.x+
- spaCy 2.x+
- NLTK 3.x+ (for now)
- numPy 1.16+
- Pandas 0.23.4+
- Future Dependencies:
    - LIWC 

All current dependencies (given that Python is installed on the device) can be installed with the following:
```
pip install xxx
```


## __Running__: 

To run the algorithm, call the following in the command line while in the repository's main directory:

```shell
python model
```
### Additional Arguments:
- ```-f ``` - Far Away Analysis
- ```-o ``` - Objectivity Analysis
- ```-v ``` : verbose


## __Algorithm__:
- The ```input\training``` directory includes the data from Experimetrix that was used to train the model.
- When the model is called, the model will automatically pull any database files from ```input\training``` and begins to fit the model based on that data. 
- If a strategy flag was included in the command line, then it will automatically learn the data on that reappraisal strategy. Otherwise, the user will be prompted to enter one. 
- Explanation of data


## __Notes__:



## __Future Updates__:
- Set logging level with -v and a number 
- Ability to specify training data
- Ability to specify testing data
- Synonym support 
- Sentiment Analysis as a token extension

