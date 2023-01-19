import numpy as np
import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns

df = pd.read_csv('fake_job_postings.csv')

# Separate majority and minority classes
df_majority = df[df.fraudulent == 0]
df_minority = df[df.fraudulent == 1]

#data exploration


#number of fake vs real job postings
sns.countplot(x='fraudulent', data = df)
plt.figure(figsize=(10,6))
plt.show()

#number of job postings by employment type
#df_majority.groupby('employment_type').fraudulent.count().plot(kind='bar', title='Real job count by employment type');
#plt.show()
#df_majority.groupby('required_education').fraudulent.count().plot(kind='bar', title='Real job count by required education');
#plt.show()





#balancing dataset


# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples=10000,     # to match minority class
                                 random_state=123) # reproducible results

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=10000,     # to match minority class
                                 random_state=123) # reproducible results

#concatenating upsampled minority and downsampled majority to create df
df = pd.concat([df_majority_downsampled, df_minority_upsampled])

#ax = sns.countplot(x='fraudulent', data=df)
#plt.show()

sns.countplot(x='fraudulent', data = df)
plt.figure(figsize=(10,6))
plt.show()

#concatenating text columns

df = df.fillna('Unspecified') 

df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department']
df['text'] = df['text'] + ' ' + df['company_profile'] + ' ' + df['description']
df['text'] = df['text'] + ' ' + df['requirements'] + ' ' + df['benefits']
df['text'] = df['text'] + ' ' + df['required_experience'] + ' ' + df['required_education']
df['text'] = df['text'] + ' ' + df['industry'] + ' ' + df['function'] + ' ' + df['employment_type']

df['text'].apply(word_tokenize)

#cleaning dataset

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

#removing stopwords
df['text'].apply(lambda x: [item for item in x if item not in sw])

def clean_text(text):

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()

    punctuation = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuation:
        text = text.replace(p,'') #Removing punctuation
        
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)

    return text

df['text'] = df['text'].apply(lambda x: clean_text(x))

#dropping unnecessary columns
df = df.drop('title', axis='columns')
df = df.drop('location', axis='columns')
df = df.drop('department', axis='columns')
df = df.drop('salary_range', axis='columns')
df = df.drop('company_profile', axis='columns')
df = df.drop('description', axis='columns')
df = df.drop('requirements', axis='columns')
df = df.drop('benefits', axis='columns')
df = df.drop('telecommuting', axis='columns')
df = df.drop('has_company_logo', axis='columns')
df = df.drop('has_questions', axis='columns')
df = df.drop('employment_type', axis='columns')
df = df.drop('required_experience', axis='columns')
df = df.drop('required_education', axis='columns')
df = df.drop('industry', axis='columns')
df = df.drop('function', axis='columns')

#printing a sample of the dataframe after cleaning
print(df)

#tfidf vectorization

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['fraudulent'],
                 test_size=0.2,random_state=123,stratify=df['fraudulent'])

tfidf_vectorizer = TfidfVectorizer() 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

tfidf_X = tfidf_vectorizer.fit_transform(df['text'])

#multinomial nb

model = MultinomialNB()
model.fit(tfidf_train, y_train)
y_pred = model.predict(tfidf_test)

#displaying nb accuracy
scores = cross_val_score(model, tfidf_X, df['fraudulent'], cv=10)
print('Train-Test Split:')
print('NB Training Accuracy: ', accuracy_score(y_train, model.predict(tfidf_train)))
print('NB Testing Accuracy: ',accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))
print('10-Fold Cross-Validation:')
print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))

#nb confusion matrix
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('NB Confusion Matrix', fontsize=18)




#knn

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(tfidf_train, y_train)
y_pred = knn.predict(tfidf_test)

#displaying knn accuracy
scores = cross_val_score(knn, tfidf_X, df['fraudulent'], cv=10)
print('\n')
print('Train-Test Split:')
print('KNN Training Accuracy: ', accuracy_score(y_train, knn.predict(tfidf_train)))
print('KNN Testing Accuracy: ',accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))
print('10-Fold Cross-Validation:')
print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))

#knn confusion matrix

conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Blues)
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('KNN Confusion Matrix', fontsize=18)
plt.show()


