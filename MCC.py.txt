import pandas as pd
import numpy as np

df = pd.read_csv('Desktop/CFD_training_set_2.csv')

import re 
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\n", "", string)    
    string = re.sub(r"\r", "", string) 
    string = re.sub(r"[0-9]", "digit", string)
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

from sklearn.model_selection import train_test_split
X = []
for i in range(df.shape[0]):
    X.append(clean_str(df.iloc[i][1]))
y = np.array(df["category"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier


model = Pipeline([('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])


from sklearn.model_selection import learning_curve, GridSearchCV
parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],
               'tfidf__use_idf': (True, False)}

gs_clf_svm = GridSearchCV(model, parameters, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X, y)
print(gs_clf_svm.best_score_)
print(gs_clf_svm.best_params_)


model = Pipeline([('vectorizer', CountVectorizer(ngram_range=(1,1))),
    ('tfidf', TfidfTransformer(use_idf=False)),
    ('clf', OneVsRestClassifier(LinearSVC(class_weight="balanced")))])

model.fit(X_train, y_train)

pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(pred, y_test)

accuracy can be found by :
accuracy_score(y_test, pred)

from sklearn.externals import joblib
joblib.dump(model, 'model_bug_classify.pkl', compress=1)

from sklearn.externals import joblib
model = joblib.load('model_bug_classify.pkl')

