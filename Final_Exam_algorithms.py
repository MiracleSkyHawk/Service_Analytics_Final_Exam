# https://docs.google.com/document/d/1nltL1z6b71elRSTwxwsoTiNdCVzvBvRu3iy4zCPTy0Y/edit#

# Packages

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from scipy.stats import t,ttest_ind,ks_2samp,shapiro,anderson 
from statsmodels.stats.weightstats import ztest
import statsmodels.api as sm
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
import os
from scipy.stats import levene,bartlett,shapiro

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import numpy as np
from autocorrect import Speller
from googletrans import Translator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter
import scipy
from scipy.spatial.distance import pdist, squareform
import sys
os.getcwd()

import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import numpy as np
from autocorrect import Speller
from googletrans import Translator, constants
from matplotlib import pyplot as plt
import re
from nltk.corpus import stopwords
from nltk import pos_tag
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import numpy as geek
import researchpy as rp
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import NearMiss
from textblob import TextBlob
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
# from wordcloud import WordCloud

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


import math 

import ciw
import matplotlib.pyplot as plt 
import pandas as pd ,seaborn as sns, numpy as np ,matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import expon, poisson,gamma
import numpy as np
from scipy.stats import ks_2samp, kstest
import statsmodels.api as sm 
warnings.filterwarnings('ignore')
np.random.seed(0)
import matplotlib 
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 15}
matplotlib.rc('font', **font)

from sklearn.model_selection import train_test_split
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,precision_recall_curve,plot_precision_recall_curve,f1_score,accuracy_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
import itertools  
from scipy.stats import mannwhitneyu
from sklearn.tree import  plot_tree 
import numpy as np
# selection of algorithms to consider and set performance measure
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split  # import 'train_test_split'
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 15}
matplotlib.rc('font', **font)
import os
from sklearn.ensemble import IsolationForest
from scipy.stats import randint as sp_randint
import warnings
warnings.filterwarnings('ignore')
os.getcwd()

################# Mann-Whitney U test ####################
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

x1 = np.array([540,670,1000,960,1200,4650,4200])
x2 = np.array([900,1300,4500,5000,6100,7400,7500])

##### Method 1 ######
# Combine the two samples
combined_sample = np.concatenate((x1, x2))

# Assign a rank to each value in the combined sample
ranks = stats.rankdata(combined_sample)

# Split the ranks into two groups based on the original samples
ranks1 = ranks[:len(x1)]
ranks2 = ranks[len(x1):]

# Calculate the U statistic for the first sample
U1 = np.sum(ranks1) - (len(x1) * (len(x1) + 1)) / 2

# Calculate the U statistic for the second sample
U2 = np.sum(ranks2) - (len(x2) * (len(x2) + 1)) / 2

# The smaller of the two U statistics is used as the test statistic for the Mann-Whitney U test
U = min(U1, U2)

print('U-Statistics is: %.3f' %(U))

##### Method 2 ######
u_statistic, p_value = stats.mannwhitneyu(x1, x2, alternative='less')
print('U-Statistics is: %.3f, P-value=%.3f' %(u_statistic,p_value))

statistic,p_value = stats.mannwhitneyu(x1, x2, alternative='two-sided')


################# T test ####################
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

##### Method 1 ######
# Calculate the sample size and mean for each sample
n1 = len(x1)
n2 = len(x2)
mean1 = np.mean(x1)
mean2 = np.mean(x2)

# Calculate the sample variance for each sample
var1 = np.var(x1, ddof=1)
var2 = np.var(x2, ddof=1)

# Calculate the pooled variance
pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

# Calculate the pooled standard deviation
pooled_std = np.sqrt(pooled_var)

t = (mean1-mean2)/(pooled_std*np.sqrt(1/n1+1/n2))

print('t-Statistics is: %.3f' %(t))

# P-value
p_value = stats.t.sf(t, df=n1+n2-2) 
print('p-value is: %.3f' %(1-p_value))

# critical value
critical_value =stats.t.ppf(0.05, n1+n2-2)
print('Critical value is: %.3f' %(critical_value))

##### Method 2 ######
statistic,p_value = stats.ttest_ind(x1,x2, alternative='less')
statistic,p_value = stats.ttest_ind(x1,x2, alternative='greater')
statistic,p_value = stats.ttest_ind(x1,x2, alternative='two-sided')
print('t-Statistics is: %.3f, P-value=%.3f' %(statistic,p_value))


################# One-way ANOVA ####################
from scipy.stats import f_oneway

data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]

stat, p = f_oneway(data1, data2, data3)


################# Levene test for variance equality ####################
from scipy.stats import levene
stat, p = levene(data1, data2, data3)

# ------------------------------------------------
# Recommendation
# ------------------------------------------------

################# Vectorization ####################
path = "C:/Users/Vahid/OneDrive - University of Toronto/MMA - 2022 -/Jupyter files/"
df_raw = pd.read_excel(path+'Data/UBER.xlsx', sheet_name ='Switchbacks' )
df = df_raw

df = (df_raw.groupby(['Order', 'Description'])['SKU'].size().unstack().reset_index().fillna(0).set_index('Order'))


################# Binarization ####################
def encode_u(x):
  if x < 1:
    return 0
  else:
    return 1

df = df.applymap(encode_u)
df.head(5)


################# Apriori Algorithm ####################
frequentitemsets = apriori(df, min_support=0.05, use_colnames=True)
frequentitemsets.sort_values(['support'],ascending=True).reset_index(drop=True)

wooden_star_rules = association_rules(frequentitemsets, metric="lift", min_threshold=1)
wooden_star_rules.sort_values(['lift','support'],ascending=False).reset_index(drop=True).iloc[:,0:7]


################# Cosine Similarity ####################
cosine_sim = 1-pairwise_distances(df,metric='cosine')
cosine_sim = pd.DataFrame(cosine_sim)


################# Dice Coefficient ####################
df_prod=df.T

a = pdist(df_prod, metric='dice')
Dice_sim_prod = pd.DataFrame(1-squareform(a))
Dice_sim_prod.columns= df.columns
Dice_sim_prod.index=df.columns
Dice_sim_prod.head(10)


# ------------------------------------------------
# Text Analytics
# ------------------------------------------------

text = """This reativly new temple's a big hindu version of Disney Land.... Quite expensive entrance for a temple. I accept the religious dress code so i was not alowed to enter in shorts. I had to rent an Indian lunghi, male long dress. The guards was not polite, almost unfriendly. One lady guard was about to destroy my expensive camera wich i had to leave outside. Inside the temple it was to follow the marked way. Somwhere you was blessed by some holy man. But everywhere there was stands with different things for sale. Worst was the way out with an enormous market with holy souvenirs of different quality, most kitscy plastic things" made in China". The holy experience i had looked forward to have dissapeared in a commercial thing. All they wanted was my money."""


################# Contraction Replacement ####################
#dictionary consisting of the contraction and the actual value
Apos_dict={"'s":" is","n't":" not","'m":" am","'ll":" will",
           "'d":" would","'ve":" have","'re":" are"}

for key,value in Apos_dict.items():
    if key in text:
        print(key)
        text=text.replace(key,value)
text

################# Regex - Remove Non-wrods ####################
text = re.sub('[^A-Za-z]+', ' ', text) 


################# Upper case to lower case ####################
text = text.lower()


################# Correct Slang ####################
path = "C:\\Users\\Vahid\\OneDrive - University of Toronto\\MMA - 2022 -\\Jupyter files\\Data\\"
#open the file slang.txt
file=open(path+"slang.txt","r")
slang=file.read()
#separating each line present in the file
slang=slang.split('\n')

# creating dictionary from slang (which is list)
Slang_dict=dict()
for line in slang:
    temp=line.split("=")
    Slang_dict[temp[0]] = temp[-1]
Slang_dict

for key,value in Slang_dict.items():
    if key in text:
        text=text.replace(key,value)
text        

################# Tokenization ####################
text = word_tokenize(text)


################# Remove stop words ####################
# Import stopwords with nltk.
stop = stopwords.words('english')
text = [word for word in text if word not in stopwords.words('english')]

lmtzr = WordNetLemmatizer()
text = ' '.join([lmtzr.lemmatize(word) for word in text])
text


################# Bag of Words - Unigram ####################
text2 = 'I went sightseeing in three people. Wide in unexpectedly, it is the most deep atmosphere in the way of local. Worship barefoot. People of shorts will visit dressed in cloth (rental). Yes Snack Corner, ants and souvenirs, is there a buffet restaurant, it has to some extent touristy, is only less expensive that the local-friendly, is heterogeneous. Worship time will take quite. Severe in 1 hour.'
word_vectorizer = CountVectorizer(ngram_range=(1,1), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform([text,text2])
df = pd.DataFrame(sparse_matrix.toarray())
df.columns = word_vectorizer.get_feature_names()
df.head()

frequencies = sum(sparse_matrix).toarray()[0]
words_count = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
words_count.sort_values(by='frequency',inplace=True,ascending=False)
words_count.head()


################# Bag of Words - Biagram ####################
word_vectorizer = CountVectorizer(ngram_range=(2,2), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform([text,text2])
df = pd.DataFrame(sparse_matrix.toarray())
df.columns = word_vectorizer.get_feature_names()
df.head()

frequencies = sum(sparse_matrix).toarray()[0]
words_count = pd.DataFrame(frequencies, index=word_vectorizer.get_feature_names(), columns=['frequency'])
words_count.sort_values(by='frequency',inplace=True,ascending=False)
words_count.head()


################# TF - IDF ####################
v = TfidfVectorizer()
sparse_matrix_tfidf = v.fit_transform([text,text2])
pd.DataFrame(sparse_matrix_tfidf.toarray(), columns=v.get_feature_names())

tfidf_weight = sum(sparse_matrix_tfidf).toarray()[0]

words_tfidf = pd.DataFrame(tfidf_weight, index=v.get_feature_names(), columns=['TF-IDF Weight'])
words_tfidf.sort_values(by='TF-IDF Weight',inplace=True,ascending=False)
words_tfidf


################# Chi-Squre Test ####################
word_vectorizer = CountVectorizer(ngram_range=(1,1), analyzer='word')
sparse_matrix = word_vectorizer.fit_transform([text,text2])
X = np.where(sparse_matrix.toarray()>=1,1,0)
X_chs = pd.DataFrame(data = X ,columns=word_vectorizer.get_feature_names_out())
X_chs['feedback']=['positive','negative']
X_chs.head()

y = X_chs['feedback']

crosstab, test_results, expected = rp.crosstab(X_chs['big'].astype(object), 
                                               y,
                                               test= "chi-square",
                                               expected_freqs= True)

crosstab
expected
test_results

# ------------------------------------------------
# Imbalance
# ------------------------------------------------

################# F1- Beta Score ####################
X, y = make_classification(n_samples=1000, n_classes=2,
                           random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.2,
                                                    random_state=2)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred_prob = lr.predict_proba(X_test)
y_pred_prob = y_pred_prob[:,1]
f1_score(y_test, y_pred)

print(fbeta_score(y_test, y_pred, beta=0.5))
print(fbeta_score(y_test, y_pred, beta=1))
print(fbeta_score(y_test, y_pred, beta=2))

_, _, threshold = precision_recall_curve(y_test, y_pred_prob)

f1score = list()
f05score = list()
f2score = list()
precision = list()
recall = list()
for th in threshold:                                                    
    y_test_pred = list()
    for prob in y_pred_prob:
        if prob > th:
            y_test_pred.append(1)
        else:
            y_test_pred.append(0)
    
    f1score.append(f1_score(y_test, y_test_pred))
    precision.append(precision_score(y_test, y_test_pred))
    recall.append(recall_score(y_test, y_test_pred))
    f05score.append(fbeta_score(y_test, y_test_pred, beta=0.5))
    f2score.append(fbeta_score(y_test, y_test_pred, beta=2))
    
_, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('Threshold')
plt.plot(threshold, precision, label='precision')
plt.plot(threshold, recall, label='recall')
plt.plot(threshold, f05score, label='F0.5')
plt.plot(threshold, f1score, label='F1')
plt.plot(threshold, f2score, label='F2')
plt.legend(loc='lower left')    


################# Random Over Sample ####################
sampler = RandomOverSampler(random_state=0,shrinkage=None)
X_res, y_res = sampler.fit_resample(X, y)
unique, counts = np.unique(y_res, return_counts=True)
print(unique, counts)


################# Random Under Sample ####################
sampler = RandomUnderSampler(random_state=0,sampling_strategy=.5)
X_res, y_res = sampler.fit_resample(X, y)
unique, counts = np.unique(y_res, return_counts=True)
print(unique, counts)


################# Smoothed Bootstrap ####################
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
kwarg_params = {'linewidth': 3}

sampler = RandomOverSampler(random_state=0,shrinkage=0)
X_res, y_res = sampler.fit_resample(X, y)
colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[0].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[0].set_title('Random Over-Sampled(Bootstrap)')

sampler = RandomOverSampler(random_state=0,shrinkage=.3)
X_res, y_res = sampler.fit_resample(X, y)
colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[1].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[1].set_title('Smoothed Bootstrap')
plt.show()


################# SMOTE ####################
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
kwarg_params = {'linewidth': 3}

colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y]
axs[0].scatter(X[:, 0], X[:, 1], c=colors, **kwarg_params)
axs[0].set_title('Original Dataset')


sampler = SMOTE(random_state=0)
X_res, y_res = sampler.fit_resample(X, y)
unique, counts = np.unique(y_res, return_counts=True)
print(unique, counts)


colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[1].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[1].set_title('SMOTE')
plt.show()


################# BorderlineSMOTE ####################
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
kwarg_params = {'linewidth': 3}

sampler = SMOTE(random_state=0)
X_res, y_res = sampler.fit_resample(X, y)
colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[0].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[0].set_title('SMOTE')

sampler = BorderlineSMOTE(random_state=0, kind="borderline-1")
X_res, y_res = sampler.fit_resample(X, y)
unique, counts = np.unique(y_res, return_counts=True)

colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[1].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[1].set_title('SMOTE borderline-1')


sampler = BorderlineSMOTE(random_state=0, kind="borderline-2")
X_res, y_res = sampler.fit_resample(X, y)
unique, counts = np.unique(y_res, return_counts=True)

colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[2].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[2].set_title('SMOTE borderline-2')

plt.show()


################# KMeans SMOTE ####################
from imblearn.over_sampling import KMeansSMOTE
from sklearn.cluster import MiniBatchKMeans
from imblearn.over_sampling import SVMSMOTE

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
kwarg_params = {'linewidth': 3}

sampler = SMOTE(random_state=0)
X_res, y_res = sampler.fit_resample(X, y)
colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[0].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[0].set_title('SMOTE')

sampler = KMeansSMOTE( kmeans_estimator=MiniBatchKMeans(n_init=1, random_state=0), random_state=0)
X_res, y_res = sampler.fit_resample(X, y)
unique, counts = np.unique(y_res, return_counts=True)

colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[1].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[1].set_title('KMeans SMOTE')


sampler = SVMSMOTE(random_state=0)
X_res, y_res = sampler.fit_resample(X, y)
unique, counts = np.unique(y_res, return_counts=True)

colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[2].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[2].set_title('SVMSMOTE')

plt.show()

################# Neasr Miss ####################
from imblearn.under_sampling import NearMiss

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 7))
kwarg_params = {'linewidth': 3}

colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y]
axs[0].scatter(X[:, 0], X[:, 1], c=colors, **kwarg_params)
axs[0].set_title('Original Observation')


sampler = NearMiss(version=1, n_neighbors=2)
X_res, y_res = sampler.fit_resample(X, y)
colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[1].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[1].set_title('NearMiss -1 ')

sampler = NearMiss(version=2, n_neighbors=2)
X_res, y_res = sampler.fit_resample(X, y)
colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[2].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[2].set_title('NearMiss -2 ')


sampler = NearMiss(version=3, n_neighbors=2)
X_res, y_res = sampler.fit_resample(X, y)
colors = ['green' if v == 0 else 'red' if v == 1 else 'black' for v in y_res]
axs[3].scatter(X_res[:, 0], X_res[:, 1], c=colors, **kwarg_params)
axs[3].set_title('NearMiss -3 ')

plt.show()


# ------------------------------------------------
# Queuing
# ------------------------------------------------

################# Helper functions ####################
def user_defined_sum(c,𝜌):
    sum = 0
    for n in range(c):
        sum += (c*𝜌)**n/factorial(n) 

    return 1/(sum + (c*𝜌)**c/(factorial(c)*(1-𝜌)))


def factorial(x):
    if (x == 1) or (x==0):
        return 1
    else:
        # recursive call to the function
        return (x * factorial(x-1)) 


################# M/M/1 ####################
def m_m_1(𝜆,µ,c=1):
    𝜌 = 𝜆/µ
    p0 = 1 - 𝜌
    lq = 𝜌**2/(1-𝜌)
    wq = lq/𝜆
    L = lq + 𝜆/µ
    W=L/𝜆
    return(𝜌,p0,lq,L,wq,W) 



################# M/M/c ####################
def m_m_c(𝜆,µ,c):
    𝜌 = 𝜆/(c*µ)
    p0 = user_defined_sum(c,𝜌)
    lq = (c*𝜌)**c*p0*𝜌/(factorial(c)*(1-𝜌)**2)
    wq = lq/𝜆
    L = lq + 𝜆/µ
    W=L/𝜆
    return(𝜌,p0,lq,wq,L,W) 


################# G/G/C ####################
def g_g_c(s,cv_s,a,cv_a,c, 𝜆):

    𝜌 = s/(c*a)
    wq = s/c * (cv_a**2+cv_s**2)/2 * 𝜌**(-1+math.sqrt(2*(c+1)))/(1- 𝜌)
    
    service_time = s 
    W = wq + s 
    L = W * 𝜆
    lq = wq * 𝜆
    return(𝜌,lq,wq,L,W)   


################# Dedicated Queues with Two Servers - Two M/M/1 ####################
𝜆 = 95/2  #arrival rate
µ = 50    # service rate 

𝜌,p0,lq,L,wq,W = m_m_1(𝜆,µ)  
print('Avg. Time in queue:{}'.format(round(wq,3)))
print('Avg. Time in system:{}'.format(round(W,3)))


################# Combined Queues - M/M/2 ####################
𝜆 = 95  #arrival rate
µ = 50    # service rate 
c =2
𝜌,p0,lq,wq,L,W = m_m_c(𝜆,µ,c) 
print('Avg. Time in queue:{}'.format(round(wq,3)))
print('Avg. Time in system:{}'.format(round(W,3)))


################# Tamdem Queues ####################
𝜆 = 95       # arrival rate
a = 1/𝜆      # inter-arrival time 
cv_a = 1  # std of inter-arrival time 

µ = 100   # service rate 
s = 1/µ   # service time 
cv_s1 = .5 # std of service time 

c = 1
𝜌,lq,wq,L,W = g_g_c(s,cv_s1,a,cv_a,c, 𝜆)
print('Avg. Time in first queue:{}'.format(round(wq,3)))
print('Avg. Time in first system:{}'.format(round(W,3)))


################# Kolmogorov-Smirnov Test ####################
# generating random varialbes from gamma distribution
𝛼 =2
𝛽 = 10
gamm_1 = stats.gamma.rvs(a=𝛼 , loc =0 , scale  =1/𝛽, size= 1000, random_state=10)

## MLE for gamma distribution
shape_gama,loc_gama ,scale_gama = stats.gamma.fit(gamm_1) 
print('Location, shape, and Scale of gamma distribution is: ', loc_gama, shape_gama,scale_gama)

# for gamma
stat, p_value = stats.kstest(gamm_1,"gamma",args=(shape_gama,loc_gama ,scale_gama)) #KS Test
if p_value<.05:
    print ('p-value is:', p_value.round(2), 'Reject H0')
else:
    print('p-value is:', p_value.round(2), 'Fail to reject H0')


################# Two M/M/1 Simulation ####################
l = 95/2 # arrival rate to each server /hour
mu = 50 # service rate  per server /hour
c = 1
decimal = 4

N = ciw.create_network(
    arrival_distributions=[ciw.dists.Exponential(rate=l)],
    service_distributions=[ciw.dists.Exponential(rate=mu)],
    number_of_servers=[1])
ciw.seed(0)

### G/M/1
#N = ciw.create_network(
#    arrival_distributions=[ciw.dists.Gamma(shape=.5, scale=1/l)], 
#    service_distributions=[ciw.dists.Exponential(rate=mu)],
#    number_of_servers=[c])
###

Q = ciw.Simulation(N, tracker=ciw.trackers.SystemPopulation())

# Stopping Criteria
#Q.simulate_until_max_customers(1000)
Q.simulate_until_max_time(24)

recs = Q.get_all_records()
df = pd.DataFrame(recs)
df.sort_values(by='arrival_date',inplace=True)
df['inter_arrival'] = df.arrival_date - df.arrival_date.shift(1,fill_value=0)
df['system_time'] = df.exit_date - df.arrival_date

df[['id_number','server_id','arrival_date','waiting_time','service_start_date','server_id','service_time','service_end_date','exit_date','queue_size_at_arrival','queue_size_at_departure']]

plt.figure(figsize=(12,4))
sns.distplot(df['inter_arrival'],kde=False,bins=20)
plt.title('Time between Arrivals Distribution')
plt.xlabel('Hours')
plt.ylabel('Frequency')
sns.despine()
plt.show()

plt.figure(figsize=(12,4))
sns.distplot(df['service_time'],kde=False,bins=20)
plt.title('Service Times Distribution')
plt.xlabel('Hours')
plt.ylabel('Frequency')
sns.despine()
plt.show()


################# Identify Steady State ####################
l = 95/2 
mu = 50 
c = 1
decimal = 4

N = ciw.create_network(
    arrival_distributions=[ciw.dists.Exponential(rate=l)],
    service_distributions=[ciw.dists.Exponential(rate=mu)],
    number_of_servers=[1])
ciw.seed(0)
Q = ciw.Simulation(N, tracker=ciw.trackers.SystemPopulation())
avg_wait_time_sim=[]


### running for different max time
for time in np.arange(1, 100,5)*24:
    Q.simulate_until_max_time(time)
    recs = Q.get_all_records()
    df = pd.DataFrame(recs)
    avg_wait_time = df['waiting_time'].mean()
    avg_wait_time_sim.append(avg_wait_time)
    
fig,ax = plt.subplots(figsize=(12,4))
ax.plot(np.arange(1, 100,5)*24, avg_wait_time_sim)
ax.set_ylabel("Average Wait Time ", fontsize=14)
ax.set_xlabel("Simultion Stop Time(Hours)", fontsize=14)
plt.legend(bbox_to_anchor=(1.6, .5))

plt.title('Steady State Behavior of Avg. Wait Time')
plt.show() 



# ------------------------------------------------
# HR Analytics
# ------------------------------------------------


################# Z Score ####################
scores = np.concatenate([np.random.normal(10, 2, size=100), np.random.normal(20, 2, size=20)])
scores[110] = 30  # add an outlier that is not extreme

# calculate the mean and standard deviation
mean = np.mean(scores)
std_dev = np.std(scores)
print(mean,std_dev)

# calculate the standard z-score
z_scores = (scores - mean) / std_dev


################# Modified Z Score ####################
# calculate the modified z-score
median = np.median(scores)
mad = np.median(np.abs(scores - median))
modified_z_scores = 0.6745 * (scores - median) / mad
print(median, mad)


################# IQR ####################
Q1,Q2,Q3 = np.percentile(scores , [25,50,75])
IQR = Q3 - Q1
lower_range = Q1 - (1.5 * IQR)
upper_range = Q3 + (1.5 * IQR)
print("Median value of Data",Q2)
print("The 25th Percentile of Data",Q1)
print("The 75th Percentile of  Data ",Q3)
print("The IQR of  Data ", IQR)
print("The Lower Range of  Data ",max(0,lower_range))
print("The Upper Range of  Data ",upper_range)

plt.boxplot(scores, whis=1.5, showfliers=True,showmeans=False,vert=False)
plt.xlabel('Data')
plt.show()
plt.show()


################# DBSCAN ####################
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)

X = StandardScaler().fit_transform(X)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10,metric='euclidean').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters - DBSCAN: %d' % n_clusters_)
plt.show()


################# Homogeneity ####################
################# Completeness ####################
################# V-measure ####################
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


X, y = make_circles(n_samples=750, factor=0.3, noise=0.1)
X = StandardScaler().fit_transform(X)
y_pred = DBSCAN(eps=0.3, min_samples=10).fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=y_pred)
print('Number of clusters: {}'.format(len(set(y_pred[np.where(y_pred != -1)]))))
print('Homogeneity: {}'.format(metrics.homogeneity_score(y, y_pred)))
print('Completeness: {}'.format(metrics.completeness_score(y, y_pred)))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))


################# Pearson Correlation ####################
from scipy import stats

# Example dataset
x = range(0,100,1)
y = np.exp(x)

# Calculate Pearson correlation coefficient and p-value
corr_pearson, p_pearson = stats.pearsonr(x, y)
print(f"Pearson correlation coefficient: {corr_pearson:.3f}, p-value: {p_pearson:.3f}")


################# Spearman Correlation ####################
from scipy import stats

# Example dataset
x = range(0,100,1)
y = np.exp(x)

# Calculate Spearman correlation coefficient and p-value
corr_spearman, p_spearman = stats.spearmanr(x, y)
print(f"Spearman correlation coefficient: {corr_spearman:.3f}, p-value: {p_spearman:.3f}")


################# Chi-Squre test of Independency ####################
from scipy.stats import chi2_contingency

# Define the contingency table
observed = [[10, 20, 30],
            [15, 25, 35],
            [5, 10, 15]]

# Calculate the chi-square test statistic, p-value, degrees of freedom, and expected frequencies
statistic, p_value, dof, expected = chi2_contingency(observed)

# Print the results
print("Chi-square test statistic:", statistic)
print("p-value:", p_value)


################# Ridge Regression ####################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

#scaler = StandardScaler()

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#X_train_scaled = scaler.fit_transform(X_train)

# define a list of alpha values to test
n_alphas = 200
alphas = [0,.0001,.001,.01,.1,1,10]

# initialize an empty list to store the accuracy scores
accuracy_values = []

# loop over the alpha values
for alpha in alphas:
    # create a Ridge classifier object with the current alpha value
    ridge = RidgeClassifier(alpha=alpha)
    
    # fit the model to the training data
    ridge.fit(X_train, y_train)
    
    # make predictions on the test data
    y_pred = ridge.predict(X_test)
    
    # compute the accuracy score and append it to the list
    accuracy_values.append(accuracy_score(y_test, y_pred))

# plot the accuracy scores against the alpha values
plt.plot(alphas, accuracy_values, '-o')
#plt.xscale('log')
plt.xlabel('$\lambda$')
plt.ylabel('Accuracy')
plt.title('Ridge classifier parameter tunning')
plt.show()


################# Gower Distance ####################
import gower
import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric

df = pd.DataFrame({'age':[14,19,10,14,21,19,30,35],
                  'pre_testscore':[4,24,31,3,3,4,31,9],
                  'available_credit':[2200,1000,22000,2100,2000,1000,6000,2200],
                  'gender':['M','M','M','M','F','F','F','F']})

##### Method 1 ######
gower.gower_matrix(df)

##### Method 2 ######
s1 = DistanceMetric.get_metric('manhattan').pairwise(df[['age']])
s1 = s1/max(np.ptp(df['age']),1)
s1

s2 = DistanceMetric.get_metric('manhattan').pairwise(df[['pre_testscore']])
s2 = s2/max(np.ptp(df['pre_testscore']),1)
s2

s3 = DistanceMetric.get_metric('manhattan').pairwise(df[['available_credit']])
s3 = s3/max(np.ptp(df['available_credit']),1)
s3

f =  df.loc[:,'gender']
dummy_df = pd.get_dummies(f)
dummy_df

s4 = DistanceMetric.get_metric('dice').pairwise(dummy_df)
pd.DataFrame(s4)

w1=w2=w3=w4=1

Gowers_Distance = (s1*w1 + s2*w2 + s3*w3+ s4*w4)/(w1 + w2 + w3+w4) 
pd.DataFrame(Gowers_Distance)


################# Classification Evaluation ####################
def evaluate(model, Y_test, X_test,X_train,Y_train): 
    
    # predict the target on the train dataset
    predict_train = model.predict(X_train)

    # Accuray Score on train dataset
    accuracy_train = accuracy_score(Y_train,predict_train)
    print('Accuracy Score on train dataset : ', round(accuracy_train,2))

    # predict the target on the test dataset
    predict_test = model.predict(X_test)

    # Accuracy Score on test dataset
    accuracy_test = accuracy_score(Y_test,predict_test)
    print('Accuracy_score on test dataset : ', round(accuracy_test,2), '\n')
    
    precision = precision_score(Y_test,  predict_test)
    recall = recall_score(Y_test,  predict_test)
    accuracy= accuracy_score(Y_test,  predict_test)
    F1_score= f1_score(Y_test,  predict_test)

    print('Model Performance')
    print('Precision: {}'.format(round(precision,2)))
    print('Recall: {}'.format(round(recall,2)))
    print('Accuracy:{}'.format(round(accuracy,2)))
    print('F1-score: : {}'.format(round(F1_score,2)),'\n')


################# Dummy Encoding ####################
data_transformed = pd.get_dummies(df, columns=df.columns[df.dtypes=='object'], drop_first=True)


################# Train Test Split ####################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


################# Decision Tree ####################
tree_1 = DecisionTreeClassifier(random_state=2)
tree_1.fit(X_train,Y_train)
evaluate(tree_1, Y_test, X_test,X_train,Y_train)


################# GridSearchCV ####################
# Create the parameter 
tree_1 = DecisionTreeClassifier(random_state=2)

param_grid = {
    'max_depth': [3, 4, 5,10],
    'min_samples_leaf': [3, 4, 5,10],
    'min_samples_split': [3, 4, 5,10,15],
    'random_state': [2]
}

# Instantiate the grid search model
tree_cv = GridSearchCV(estimator = tree_1, param_grid = param_grid, scoring = 'recall', cv = 10, n_jobs = -1, verbose = 0)
tree_cv.fit(X_train, Y_train)
evaluate(tree_cv, Y_test, X_test,X_train,Y_train)


################# Random Forest ####################
forest = RandomForestClassifier(random_state=2)
forest.fit(X_train, Y_train)

evaluate(forest, Y_test, X_test,X_train,Y_train)


################# Random Search ####################
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

n_iter_search = 20
random_search = RandomizedSearchCV(forest, param_distributions=param_dist,
                                   n_iter=n_iter_search)

random_search.fit(X_train, Y_train)
evaluate(random_search, Y_test, X_test,X_train,Y_train)

best_params = random_search.best_params_
print(f"Best paramters Random Search: {best_params})")


################# Bayesian Search ####################
import skopt
from skopt import BayesSearchCV

param_dist =  {
        'n_estimators': (5,100),
        'max_features': ['auto','sqrt'],
        'max_depth': (2,20),
        'min_samples_split': (2,10),
        'min_samples_leaf': (1,7),
        'bootstrap': ["True","False"]
    }

search = BayesSearchCV(estimator=forest, search_spaces=param_dist, n_jobs=-1, n_iter=32,cv=3, scoring='roc_auc')
search.fit(X_train, Y_train)
print(search.best_score_)
print(search.best_params_)
evaluate(search, Y_test, X_test,X_train,Y_train)


################# Gradient Boosting ####################
gbm = GradientBoostingClassifier(n_estimators=100,random_state=2)
gbm.fit(X_train, Y_train)
evaluate(gbm, Y_test, X_test,X_train,Y_train)


################# XGBoost (eXtreme Gradient Boosting) ####################
xgbc = XGBClassifier(random_state=2)
xgbc.fit(X_train, Y_train)
evaluate(xgbc, Y_test, X_test,X_train,Y_train)


################# Ada Boosting ####################
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()

ada.fit(X_train, Y_train)
evaluate(ada, Y_test, X_test,X_train,Y_train)


ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

parameters = {'base_estimator__max_depth':[i for i in range(2,11,2)],
              'base_estimator__min_samples_leaf':[5,10],
              'n_estimators':[10,50,250,1000],
              'learning_rate':[0.01,0.1]}

ada_cv = GridSearchCV(ada, parameters,verbose=0,scoring='f1',n_jobs=-1)
ada_cv.fit(X_train,Y_train)
evaluate(ada_cv, Y_test, X_test,X_train,Y_train)


################# Ensemble Stacking  ####################
from sklearn.ensemble import StackingClassifier

level0 = list()
level0.append(('Ada', ada_cv))
level0.append(('XGBOOST', xgbm_cv))
level0.append(('GBM', gbm_cv))
level0.append(('RF', forest_cv))
level0.append(('Tree', tree_cv))
level1 = LogisticRegression()
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
model.fit(X_train,Y_train)
evaluate(ada_cv, Y_test, X_test,X_train,Y_train)


################# Feature Importance  ####################
k=data_transformed.drop(['Attrition_Yes'], axis=1).columns

feature_imp_xgb = pd.Series(xgbm_clf.feature_importances_,index=k).sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=feature_imp_xgb[0:10], y=feature_imp_xgb.index[0:10])
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Top 10 Important Features using XGBOOST")
plt.legend()
plt.show()




