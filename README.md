# Linguistic Distancing Model 

## __Dependencies__ :
EDIT 3/19/2020 - Removed ```venv``` support in favor of ```pipenv``` in order to track dependencies without having to generate a ```venv``` folder.

Instructions for installation of ```pipenv``` are located [here](https://pipenv.kennethreitz.org/en/latest/install/#installing-pipenv).

Starting up the virtual environment and install all dependencies:
```shell
pipenv --python /usr/bin/python3
```



## __Running__: 

To run the algorithm, call the following in the command line while in the repository's main directory:

```shell
python model -args
```
### Additional Arguments:
- ```-f ``` - Far Away Analysis - Flag to enable analysis for far away/spatiotemporal distancing. 
- ```-o ``` - Objectivity Analysis - Flag to enable analysis for objective distancing. 




## __Algorithm__:
- The ```input\training``` directory includes the data from Experimetrix and LDH Phase 1 that was used to train the model. See "Data Format" to see the formatting of the training data.
- When the model is called, the model will automatically pull any database files from ```input\training``` and begins to fit the model based on that data. 
- If a strategy flag was included in the command line, then it will automatically learn the data on that reappraisal strategy. Otherwise, the user will be prompted to enter one. 
- After training, the user will be prompted via file dialog to select a table of test data. The test data should be formatted similarly to the formatting shown below. 

### __Model Fitting__:

To fit the model, the model first predicts the part-of-speech tag for each word in each response using SpaCy. Then, two things happen:

#### Score Extrapolation:
For each response, each word-tag pair is classified into specific predetermined categories, and the model counts the number of positive category matches and the number of negative category matches. From these counts and the raw score of the response, the model extrapolates a positive score and a negative score, which serve as a base score. The extrapolation of this score satisfies two qualities:
- The sum of all scores, positive and negative, equals the raw response score.
- All positive category matches contribute an equal amount of positive score, and all negative category matches contribute an equal amount of negative score. 

The model then applies those scores to each word-tag pair and also applies the score to each category. 

#### Weight Extrapolation:
The number of categories is counted 

After applying this process to every response, the model then transforms the word-tag pair into a tag->word->score mapping, where the score is an average of all occurences of word-tag scores. The model also performs a Least Squares Best Fit on the weights to get the best fit of each category. 

### __Model Prediction__:
To predict the score of text, the model tags the input text and then simply looks up the score of each word based on the trained tag-word dictionary. If the word is a novel word (thus not in the dictionary), the model searches for synonyms of that word within the mapping and finds the average of all the synonym's scores. Once this is found, the dictionary is updated with that word-tag's score. 

### _The Sentiment Problem_:
A current problem we are still trying to solve is the best way to incorporate sentiment into objective analysis; by definition, objectivity is the lack of emotionally charged language. 
Below is an analysis of each way objective analysis is run with regards to sentiment. All sentiment scores are generated with the following algorithm:

- Run each word through TextBlob in order to obtain sentiment scores, which contain a score for __subjectivity__ (between 0 and 1) and a score for __polarity__ (between -1 and 1). 
- Transform the polarity score as such:
```
new_polar = [-1 * abs(old_polar)] + 1
```
- Add a trivial amount if the score is 0 to avoid divide by zero errors (0.0001).

This means that texts that are more subjective and less polar have higher scores and less subjective/more polar scores have lower scores, on a scale of 0 to 1. 


#### Base: Objective Analysis without Sentiment
Objective (No Sentiment) - 0.09807345

#### Sentiment as a Scale Factor:  	 

This method applies sentiment changes as a multiplier after all the words have been scored. This application can occur at the sentence level, where the score of the entire sentence is scaled by the sentiment of the entire sentence before the words are scored, and/or at the word level, where each word's score is scaled by its sentiment after the word has been been scored. The table below shows the correlation coefficient of models that incorporate different parts of sentiment at different stages of the word fitting. 

| Method | Subjectivity  | Polarity | Both | 
| ------ | -------- | ----------------- | -------------- |
| Sentence Level | 0.18614455   | 0.01918858  | 0.145313 | 
| Word Level |	-0.0409038 |	0.17185006 |	0.154954 |
|Both |	-0.0983903 |	0.02679118 |	0.109418 | 

Note that these scores take into account sentiment during prediction as well, which doesn't make a whole lot of sense; check back in to see sentiment updates with those removed.

### __The Sentiment Problem__ (continued):
We realized that the subjectivity score was also supposed to be transformed. 
Since we changed the scaling on the subjectivity, we also just started calling it objectivity lol. 

Sentiment as a Scale Factor:  	 

| Method            | Objectivity  | Polarity | Both | 
| ------            | -------- | ----------------- | -------------- |
| Sentence Level    | 0.10045464 | 0.12799273   | 0.13602715 | 
| Word Level        | 0.26252657 | 0.17185006   | 0.17812991 |
| Both              | 0.29619390 | 0.17131215   | 0.18331030 | 


#### Sentiment as a Weighted Category:

This method considers sentiment as its own principle component, where subjectivity and polarity are both weights. Note that, in this framing, both polarity and subjectivity is considered a positive category (more subjective/less polar words have greater scores and thus contribute more of a positive score to the total response distancing). See branch `sentiment_change` for exact changes. As this is a new path we're looking at, it's likely that the changes may not be completely sound for our use case, but the preliminary results are below. 

| Method | Score | 
| ------ | -------- | 
| Polarity and Subjectivity | 0.03153471773761692   |
| Polarity Only |	0.15093689008145939 |
| Subjectivity Only |	-0.11981205316723109 |

### __The Sentiment Problem__ (continued):
We realized that the subjectivity score was also supposed to be transformed. 
Since we changed the scaling on the subjectivity, we also just started calling it objectivity lol. 

Sentiment as a Scale Factor:  	 

| Method            | Objectivity  | Polarity | Both | 
| ------            | -------- | ----------------- | -------------- |
| Sentence Level    | 0.10045464 | 0.12799273   | 0.13602715 | 
| Word Level        | 0.26252657 | 0.17185006   | 0.17812991 |
| Both              | 0.29619390 | 0.17131215   | 0.18331030 | 

## __Data Format__:

| Response | Objectivity Score | Far Away Score | 
| -------- | ----------------- | -------------- |
| A utf-8 string of the sentence being analyzed. | The average score, represented as a float, given by human coders for the corresponding sentence for objectivity. | The average score, represented as a float, given by human coders for the corresponding sentence for far away distancing. | 



## __Notes__:



## __Future Updates__:


