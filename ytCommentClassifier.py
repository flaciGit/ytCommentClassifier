'''
Youtube comment spam classifier,
Using TFIDF vectorization and Multi-layer Perceptron trained on data from
https://archive.ics.uci.edu/ml/index.php

Fetching live comments from a youtube video and classifying them with the model
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics;
from sklearn.neural_network import MLPClassifier

from nltk.corpus import stopwords

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pafy


API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'
DEVKEY = 'YOUR_DEVKEY'

#avicii
videoId = '1oAn81h21aQ'
url = 'www.youtube.com/watch?v=1oAn81h21aQ'
#rihanna
videoId = 'lWA2pjMjpBs'
url = 'www.youtube.com/watch?v=lWA2pjMjpBs'

video = pafy.new(url)
pafy.set_api_key("YOUR_APIKEY")
                 
youtube = build(API_SERVICE_NAME, API_VERSION,developerKey=DEVKEY)


results = youtube.commentThreads().list(
		    part="snippet",
		    maxResults=100,
		    videoId=videoId,
		    textFormat="plainText"
		  ).execute()
totalResults = 0
totalResults = int(results["pageInfo"]["totalResults"])

count = 0
nextPageToken = ''
comments = []
processingData = True
first = True
commentLimit = 5000
commentLimit = 2000
maxResults = 100

print("START")

print("Fetching data")
while processingData:
    
    error = False
    if first == False:
        print(".",end =" ")
        try:
            results = youtube.commentThreads().list(
	  		  part="snippet",
	  		  maxResults=maxResults,
	  		  videoId=videoId,
	  		  textFormat="plainText",
	  		  pageToken=nextPageToken
	  		).execute()
            totalResults = int(results["pageInfo"]["totalResults"])
        except HttpError as e:
            print("An HTTP error %d occurred:\n%s" % (e.resp.status, e.content))
            error = True

    if not error:
        count += totalResults
        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]
            author = comment["snippet"]["authorDisplayName"]
            text = comment["snippet"]["textDisplay"]
            #comments.append([author,text])
            comments.append(text)
            
        if totalResults < 100:
            processingData = False
            first = False
        else:
            processingData = True
            first = False
        try:
            nextPageToken = results["nextPageToken"]
        except KeyError as e:
            print("An KeyError error occurred: %s" % (e))
            processingData = False
                
    if count > commentLimit - maxResults:
        break
     
print()

# local classified data
input_file = "Youtube04-Eminem.csv"

data = pd.read_csv(input_file, header = 0,usecols = ["CONTENT", "CLASS"])

X_train, X_test = train_test_split(data, test_size=0.4, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english',max_df=0.8,min_df=0.05)

X = vectorizer.fit_transform(X_train.CONTENT)
Y = vectorizer.transform(X_test.CONTENT)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)

clf.fit(X,X_train.CLASS)
ds_train_pred = clf.predict(X);
ds_test_pred = clf.predict(Y);

print(metrics.classification_report(X_test.CLASS, ds_test_pred,
                                    target_names=["Nem Spam", "Spam"]));

# building result table from test and prediction on tests
result = pd.DataFrame(data = [np.array(X_test.CONTENT),
                              np.array(X_test.CLASS),
                              ds_test_pred]);
result = result.T;
result.columns = ['Comment', 'Spam', 'Spam Prediction'];

# apply new comment for the model

comments_preproc = pd.DataFrame(comments,columns = ['Comment'])

# stopword removal

# first download
# nltk.download('stopwords')

stop = stopwords.words('english')
comments_preproc['Comment'] = comments_preproc['Comment'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

'''
# punctuation removal
comments_preproc['Comment'] = comments_preproc['Comment'].apply(lambda x: " ".join(x for x in x.split() if x not in string.punctuation))

# filter non alphabetical and numreical characters
comments_preproc['Comment'] = comments_preproc['Comment'].apply(lambda x: " ".join(x for x in x.split() if x.isalnum() and not x.isdigit()))
'''

Y = vectorizer.transform(comments_preproc['Comment'])
comments_pred = clf.predict(Y);

result = pd.DataFrame(data = [comments,comments_preproc['Comment'],comments_pred]);
result = result.T;
result.columns = ['Comment','Preprocessed Comments', 'Spam Prediction'];

print("END")
