import pandas as pd

import ray
# import modin.pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt


from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 8

from scipy.stats import norm
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar

import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# nltk.download('punkt')
# nltk.download('vader_lexicon')

ray.init()


# df_train = pd.read_csv('https://raw.githubusercontent.com/harditsingh/IntroDataScience/master/Kickstarter%20Project/KS_train_data.csv', delimiter=',')

df_test = pd.read_csv('./KS_train_data.csv', delimiter=',')

# Simple cleanup
# df_train=df_train[df_train['blurb'].notnull()]
# df_train=df_train[df_train['country'].notnull()]
# df_train=df_train[df_train['name'].notnull()]

df_test=df_test[df_test['blurb'].notnull()]
df_test=df_test[df_test['country'].notnull()]
df_test=df_test[df_test['name'].notnull()]



df_test['compound'] = 0.0
df_test['neu'] = 0.0
df_test['neg'] = 0.0
df_test['pos'] = 0.0

# df_test = df_test[0:10000]
# print(df_test['blurb'])

@ray.remote
def sentiment_analysis(blurb, index):
    # Empty list is created for storing the results of the next segment
    # print(index, " ", blurb)
    # analyzed_sentences = []

    # We loop through all the reviews that we import from the file
    
    # A dictionary is created to store the data of one sentence temporarily
    data = {'compound': 0, 'neu': 0, 'neg': 0, 'pos': 0}

    # Reviews are taken, one at a time, from the review texts list
#     blurb = df['blurb'][index]
    # And then the review is separated into sentences
    sentence_list = nltk.tokenize.sent_tokenize(blurb)

    # Then, Vader Analyzer from the NLTK Library is used to do a sentiment analysis of each of the sentences obtained
    #  from the review. This analyzer gives us four parameters in the result: Compound, Neutral, Positive and Negative
    vader_analyzer = SentimentIntensityAnalyzer()
    for text in sentence_list:
        temp = vader_analyzer.polarity_scores(text)
        for key in ('compound', 'neu', 'neg', 'pos'):
            # Here, an average of the parameters is taken for all the sentences obtained from the review to find the
            # Vader Analysis scores for the review
            if sentence_list.__len__() is not 0:
                data[key] += temp[key]/sentence_list.__len__()


    # We add all the analysis scores in a list for later use


    return (index, data)


futures = [sentiment_analysis.remote(df_test['blurb'][index], index) for index in df_test.index]
final = ray.get(futures)
print(ray.get(futures))
print("done")

# @ray.remote
# def addData(index, final):
#     # print("Data: ", df_test.loc[index])
#     # if(index != final[0]):
#     #     # print("Index: ", index, ", Data: ", final)
#     #     print("what")
#     df_test['compound'][index] = final[1]['compound']
#     df_test['neu'][index] = final[1]['neu']
#     df_test['neg'][index] = final[1]['neg']
#     df_test['pos'][index] = final[1]['pos']

# temp = [addData.remote(index, final[index]) for index in df_test.index]
# temp1 = ray.get(temp)

# print(final[:][1].values())

compound = []
neu = []
neg = []
pos = []

for item in final:
    compound.append(item[1]['compound'])
    neu.append(item[1]['neu'])
    neg.append(item[1]['neg'])
    pos.append(item[1]['pos'])

# for index in df_test.index:
#     print("Index: ", index, ", Data: ", final[index][1]['compound'])
#     if(index != final[index][0]):
#         print("Index: ", index, ", Data: ", final[index])
#         print("what")
#     df_test['compound'][index] = final[index][1]['compound']
#     df_test['neu'][index] = final[index][1]['neu']*100
#     df_test['neg'][index] = final[index][1]['neg']
#     # df_test['pos'] = index
#     df_test['pos'][index] = final[index][1]['pos']

# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0.12]

df_test['pos'] = pos
df_test['neg'] = neg
df_test['neu'] = neu
df_test['compound'] = compound

# print(df_test)
df_test.to_csv('cleaned_train.csv', index=False, sep=';')