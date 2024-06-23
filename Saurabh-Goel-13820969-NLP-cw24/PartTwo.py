import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report

data = pd.read_csv(Path.cwd() / "p2-texts"/"hansard40000.csv")
print(data.head())

###Replacing 'Labour (Co-op) with Labour'

data = data.replace({'party': {'Labour (Co-op)' : 'Labour'}})
print(f"Unique party values in data before keeping only top 4 parties: {data['party'].unique()}")

## Get Top 4 parties by count excluding 'Speaker'

top_4_parties = pd.DataFrame(data.value_counts('party')).reset_index().sort_values(by = 'count', ascending=False)
top_4_parties_list = top_4_parties.loc[top_4_parties['party'] != 'Speaker']['party'][0:4].tolist()
## Removing rows where party value not in top_4_parties_list

data = data[data['party'].isin(top_4_parties_list)]
print(f"Unique party values in data after keeping only top 4 parties: {data['party'].unique()}")

### Remove any row where speech_class column value != 'Speech'

data = data.loc[data['speech_class'] == 'Speech']
print(f"Unique values in Speech Classes after dropping other speech classes: {data['speech_class'].unique()}")
print(f"Shape before dropping rows based on speech length: {data.shape}")

### Remove any row where speech length is less than 1500 characters

data['num_char'] = data['speech'].map(lambda x: len(x))
data = data.loc[data['num_char'] >= 1500].reset_index(drop = True)
data = data.drop(columns='num_char')
print(f"Shape after dropping rows based on speech length: {data.shape}")

#import nltk
# nltk.download('stopwords')
stop = list(set(stopwords.words('english')))


### Defining TfiffVectorizer with english stop words and max features = 4000
#### Defining a function to be called for uni, bi and tri grams

def vectorizer(text_data, min_grams = 1, max_grams = 1, stopwords = stop, max_features = 4000):
    tf = TfidfVectorizer(stop_words=stopwords, max_features=max_features, ngram_range=(min_grams, max_grams))
    text_tf = tf.fit_transform(text_data)
    return text_tf

### Defining function for RandomForestClassifer - Output Prediction

def rf_classier(n_estimators, X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return y_pred


### Defining function for Linear SVC - Output Prediction

def svclassifier(kernel, X_train, y_train, X_test):
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    return y_pred

### Defining function to get F1 Score and Classification Report
def classification_metrics(y_test, y_pred, target_names):
    f1 = f1_score(y_test, y_pred, average='macro')
    class_report = classification_report(y_test, y_pred, target_names=target_names)
    return f1, class_report

#### Unigram - TfidfVectorizer
speech_tf = vectorizer(data['speech'], 1, 1, stop, 4000)
## Splitting the data and into test and train with test sample = 30%
X_train, X_test, y_train, y_test = train_test_split(speech_tf, data['party'], test_size=0.3, random_state=99)
## Calling Random Forest Calssifier
y_pred_rf = rf_classier(400, X_train, y_train, X_test)
f1_score_rf, classification_report_rf = classification_metrics(y_test, y_pred_rf, top_4_parties_list)
print(f"F1-Score for Random Forest Classifier - Unigram: {f1_score_rf}")
print(f"Classification Report for Random Forest Classifier - Unigram: {classification_report_rf}")
## Calling Linear SV Classifier
y_pred_svc = svclassifier('linear', X_train, y_train, X_test)
f1_score_svc, classification_report_svc = classification_metrics(y_test, y_pred_svc, top_4_parties_list)
print(f"F1-Score for SVM Classifier - Unigram: {f1_score_svc}")
print(f"Classification Report for Linear Support Vector Classifier - Unigram: {classification_report_svc}")
#############################

#### Bigram - TfidfVectorizer
speech_tf = vectorizer(data['speech'], 2, 2, stop, 4000)
## Splitting the data and into test and train with test sample = 30%
X_train, X_test, y_train, y_test = train_test_split(speech_tf, data['party'], test_size=0.3, random_state=99)
## Calling Random Forest Calssifier
y_pred_rf = rf_classier(400, X_train, y_train, X_test)
f1_score_rf, classification_report_rf = classification_metrics(y_test, y_pred_rf, top_4_parties_list)
print(f"F1-Score for Random Forest Classifier - Bigram: {f1_score_rf}")
print(f"Classification Report for Random Forest Classifier - Bigram: {classification_report_rf}")
## Calling Linear SV Classifier
y_pred_svc = svclassifier('linear', X_train, y_train, X_test)
f1_score_svc, classification_report_svc = classification_metrics(y_test, y_pred_svc, top_4_parties_list)
print(f"F1-Score for SVM Classifier - Bigram: {f1_score_svc}")
print(f"Classification Report for Linear Support Vector Classifier - Bigram: {classification_report_svc}")
#############################

#### Trigram - TfidfVectorizer
speech_tf = vectorizer(data['speech'], 3, 3, stop, 4000)
## Splitting the data and into test and train with test sample = 30%
X_train, X_test, y_train, y_test = train_test_split(speech_tf, data['party'], test_size=0.3, random_state=99)
## Calling Random Forest Calssifier
y_pred_rf = rf_classier(400, X_train, y_train, X_test)
f1_score_rf, classification_report_rf = classification_metrics(y_test, y_pred_rf, top_4_parties_list)
print(f"F1-Score for Random Forest Classifier - Trigram: {f1_score_rf}")
print(f"Classification Report for Random Forest Classifier - Trigram: {classification_report_rf}")
## Calling Linear SV Classifier
y_pred_svc = svclassifier('linear', X_train, y_train, X_test)
f1_score_svc, classification_report_svc = classification_metrics(y_test, y_pred_svc, top_4_parties_list)
print(f"F1-Score for SVM Classifier - Trigram: {f1_score_svc}")
print(f"Classification Report for Linear Support Vector Classifier - Trigram: {classification_report_svc}")
#############################

#### Uni, Bi and Trigram - TfidfVectorizer
speech_tf = vectorizer(data['speech'], 1, 3, stop, 4000)
## Splitting the data and into test and train with test sample = 30%
X_train, X_test, y_train, y_test = train_test_split(speech_tf, data['party'], test_size=0.3, random_state=99)
## Calling Random Forest Calssifier
y_pred_rf = rf_classier(400, X_train, y_train, X_test)
f1_score_rf, classification_report_rf = classification_metrics(y_test, y_pred_rf, top_4_parties_list)
print(f"F1-Score for Random Forest Classifier - Uni, Bi and Trigram: {f1_score_rf}")
print(f"Classification Report for Random Forest Classifier - Uni, Bi and Trigram: {classification_report_rf}")
## Calling Linear SV Classifier
y_pred_svc = svclassifier('linear', X_train, y_train, X_test)
f1_score_svc, classification_report_svc = classification_metrics(y_test, y_pred_svc, top_4_parties_list)
print(f"F1-Score for SVM Classifier - Uni, Bi and Trigram: {f1_score_svc}")
print(f"Classification Report for Linear Support Vector Classifier - Uni, Bi and Trigram: {classification_report_svc}")
#############################


