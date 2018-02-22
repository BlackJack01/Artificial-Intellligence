import pandas as pd
train= pd.read_csv('E:\\Machine_Learning\\Data\\IMDB_data\\labeledTrainData.tsv', header=0, delimiter = '\t', quoting =3)

from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocessing_per_review(review):
    without_html_tags=BeautifulSoup(review)
    letters_only=re.sub('[^a-zA-Z]', ' ', without_html_tags.get_text())
    lower_case=letters_only.lower()
    words=lower_case.split()
    words=[w for w in words if not w in stopwords.words('english')]
    return ' '.join(words)
    
def preprocessing_reviews(reviews):
    processed=[]
    for i in range(len(reviews)):
        processed.append(preprocessing_per_review(reviews[i]))
    return processed

processed_reviews=preprocessing_reviews(train['review'])

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
train_data=vectorizer.fit_transform(processed_reviews)

train_data=train_data.toarray()

vocab=vectorizer.get_feature_names()

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=100)
model=forest.fit(train_data, train['sentiment'])

test= pd.read_csv('E:\\Machine_Learning\\Data\\IMDB_data\\testData.tsv', header=0, delimiter = '\t', quoting =3)
test_processed_reviews=preprocessing_reviews(test['review'])
test_data=vectorizer.transform(test_processed_reviews)
test_data=test_data.toarray()
predicted_sentiments=model.predict(test_data)
