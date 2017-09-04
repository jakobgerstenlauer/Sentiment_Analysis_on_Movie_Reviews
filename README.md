# Sentiment Analysis based on Movie Reviews using Recurrent Neural Networks (LSTM and PLSTM)

### Dependencies 
python 2.7
numpy
gensim 
nltk 
panda 
keras

### Usage:
After making the script plstm_validation.py executable it can be called with the following flags:
`./plstm_validation.py -d <DROP_OUT_RATIO> -e <MAX_EPOCHS>`

For more information type:
`./plstm_validation.py -h`

### Background Information

Kaggle hosted a sentiment analysis competition in February of 2014 for the machine learning community to benchmark their ideas using the Rotten Tomatoes movie review dataset (a corpus of movie reviews). The goal was to label phrases on a scale of five: negative, somewhat negative, neutral, somewhat positive, positive. This task is difficult because of negation, sarcasm, terseness, language ambiguity. As a result, the order of words and sentence structure must be taken into account.

### Technical details

The dataset was divided into a training data (80%) and validation data (20%) set.
Each subset was pre-processed:
+ Tokenize each row.
+ Remove all stopwords.
+ Do stemming (SnowballStemmer).
+ Create a global dictionary with all (stemmed) words from the train and test set.
+ Associate each word in this dictionary with an unique id (Single tokens were classified with the label 2).
+ Create a sequence of token ids for both test and train set.
+ Create a matrix of binary dummy variables for the labels (one dummy variable for each category/ one-hot encoding).
  
The training data consisted of the following:
 
+ Dictionary: 13,759 different tokens (both training and test)
+ There are 8,544 sentences
+ There are 156,060 sentences, phrases and single tokens
+ Labels for each of the types above
  
The test data consisted of the following:
 
+ Dictionary: 13,759 tokens (both training and test)
+ There are 3,311 sentences
+ There are 66,292 sentences, phrases and single tokens
 
After some experimentation with different layer layouts and PLSTM, the following model has been used:
 
{Embedding-layer,LSTM-layer,Dense-layer,Softmax-layer}
