# classifier.py
"""

CS 373
What's Cooking?

Tanner Ward - ward166@purdue.edu
Liam Brown - brow1368@purdue.edu
Nathaniel Young - young410@purdue.edu
Anthony Ralston - aralsto@purdue.edu

"""

# Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

# Read json data & prepare dataframe
print("Reading json data & preparing dataframes...")
training_dataframe = pd.read_json('train.json').set_index('id')
testing_dataframe = pd.read_json('test.json').set_index('id')
training_df_index = training_dataframe.index
testing_df_index = testing_dataframe.index
y = training_dataframe.cuisine.copy()
print("Training Data Samples: ", training_dataframe.shape)
print("Testing Data Samples: ", testing_dataframe.shape)

# Combine for pre-processing
print("Combining data for pre-processing...")
dataframe = pd.concat([training_dataframe.drop('cuisine', axis=1), testing_dataframe])
dataframe_index = dataframe.index
print("Concatenated Samples: ", dataframe.shape)

# Visualise cuisine training data
print("Displaying viualisation of cuisine training data...")
#sns.countplot(y=y_training, order=y_training.value_counts().reset_index()['index'])
#plt.title("Cuisine Distribution")
#plt.show()

# Apply count vectorizer
print("Applying count vectorizer...")
countVect = CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], lowercase=False)
word_matrix = countVect.fit_transform(dataframe['ingredients'].apply(','.join)).todense()
feature_names = countVect.get_feature_names()
dataframe = pd.DataFrame(word_matrix, columns=feature_names)
dataframe.index = dataframe_index

# Modeling preparation
print("Preparing for modeling...")
X = dataframe.loc[training_df_index,:]
testing_dataframe = dataframe.loc[testing_df_index,:]
print("Number of Cuisine Types: ", y.nunique())
print("Training Dataset Shape: ", X.shape)
print("Testing Dataset Shape: ", testing_dataframe.shape)

# Training modeling
"""
Logistic Regression model provided out of the box has a
number of parameters. See: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.

Reviewing the parameters, it seems we should set the following:
- Penalty: Penalty can be 'l1' or 'l2'. Given a count vectorizer has been used and feature weights are
            either 0 or 1, there shouldn't be much difference between 'l1' and 'l2' regularization.
- Solver/Multi_Class: 'ovr' appears to be the appropriate multi_class option. OVR or One vs. Rest
                        handles multi-class problems by training a single classifier for each class
                        with the samples from that class positive and the samples from all other classes
                        negative.
                        When selecting 'ovr', 'liblinear' needs to be selected for solver.
"""
print("Training model...")
model = LogisticRegression(penalty='l2', solver='liblinear', multi_class='ovr')
model.fit(X, y)

print("Predicting...")
y_pred = model.predict(testing_dataframe)
y_pred_dataframe = pd.Series(y_pred, index=testing_df_index).rename('cuisine')
print(y_pred_dataframe)
