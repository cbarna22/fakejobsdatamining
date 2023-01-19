# fake-jobs-datamining
Detecting Fake Job Ads Using Datamining Techniques

In this project, I built a script to detect whether an online job advertisement is real or fraudulent using Multinomial Naive Bayes and K-Nearest Neighbors algorithms. Employment scams are a serious issue which can result in applicants unintentionally giving their personal information to bad actors and potentially losing money. The job advertisement data is sourced from https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction. Dataset columns include job title, location, department, description, requirements, required experience, and more. 

This Python script uses the libraries numpy, pandas, re, string, nltk, sklearn, matplotlib, and seaborn. After importing various packages from these libraries, I created a pandas dataframe and imported the job ad data. Since the data was severely unbalanced (there were far more real jobs in the dataset than fake jobs), I downsampled the number of real jobs and upsampled the number of fake jobs in order to reduce bias. I replaced any null values with the string 'Unspecified'. Next, I concatenated all of the text columns of the dataset into a single column named 'text'. This column is what I used for datamining. 

The next step was to clean the data, which included converting the text to lowercase, removing stopwords/punctuation, and lemmatizing the text. After this, I transformed the data using TF-IDF vectorization. Lastly, I fed the data into the Naive Bayes and K-Nearest Neighbors models and assessed their accuracy.
