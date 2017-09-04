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
+ Tokenize each row
+ Remove all stopwords
+ Do stemming (SnowballStemmer)
+ Create an id for each token
+ Connect each id with the correct label
+ One-hot encoding
 
The preprocessed data is an arrays of length 12. Each element corresponds to a token, with a number with the token-id or a zero (which means that there is no token there). In this way each sentence that is being classified is a vector consisting of tokens that each build a sentence. The zero-padding is from left to right, so that the vector contains the information in the end, which benefits the LSTM-/PLSTM-layers. The label for each vector is represented as a 5-element binary vector (one-hot encoding). The input training data consists of the following:
 
+ Dictionary: 13759 different tokens (both training and test)
+ There are 8544 sentences
+ There are 156060 sentences, phrases and single tokens
+ Labels for each of the types above
 
The input test data consists of the following:
 
+ Dictionary: 13759 tokens (both training and test)
+ There are 3311 sentences
+ There are 66292 sentences, phrases and single tokens
 
Single tokens are classified with the label 2. The output data is the classified label.
 
The models that have been used are 4 different models, where each model has been tested with different hyperparameters. The models that have been built are the following:
 
{Embedding-layer,LSTM-layer,Dense-layer,Softmax-layer}

{Embedding-layer,PLSTM-layer,Dense-layer,Softmax-layer}

{Embedding-layer,LSTM-layer,LSTM-layer,Dense-layer,Softmax-layer}

{Embedding-layer,PLSTM-layer,PLSTM-layer,Dense-layer,Softmax-layer}
 
The objective for calculation of the loss that has been used is the categorical crossentropy, also known as the multiclass logloss. We also shuffle the data-samples for each epoch.
 
The both optimizers uses moving average which, compared to SGD, allows the algorithms to take bigger steps and therefore they converge faster.
