import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #(funkcja)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV #(klasa)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv("bbc_news.csv")

#zamaina labeli ze stringow na int
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df['type'])

df['label'] = label_encoder.transform(df['type'])

#wymieszanie rekordow w df
df = df.sample(frac = 1).reset_index(drop = True)

#rozdzielenie df na 2 podzbiory (treningowy i testowy w stosunku 70%:30%)
train, test = train_test_split(df, test_size = .3)

#pipeline to kroki - preprocessing, uczenie i walidajca - uruchamiane jedna komenda
pipeline = Pipeline([
    ('vectorization', CountVectorizer(lowercase = True, min_df=2, stop_words = stopwords.words('english') )),
    ('classification', MultinomialNB())
])

#grid search przeiteruje sie po wszystkich kombinajcach parametrow i wybierze najlepsze
parameters = {
    'vectorization__min_df': (2,3)
}

#losowanie wszystkich parametrow, uczenie
grid_search = GridSearchCV(pipeline, parameters)

grid_search.fit(train['news'], train['label'])

#Rezultat
print('Best score: %.4f' % grid_search.best_score_)
print(grid_search.best_params_)

y_true, y_pred = test['label'], grid_search.predict(test['news'])

print(classification_report(y_true, y_pred))

#Klasyfikacja
#Real life example
print(label_encoder.inverse_transform(grid_search.predict(['Your mum sucks'])))

#Test data
for x in range(len(test)):
  predicted = label_encoder.inverse_transform(grid_search.predict([test.iloc[x]['news']]))
  if predicted[0] != test.iloc[x]['type']:
    print("Prediction/label: {}/{},  News: {}".format(predicted[0], test.iloc[x]['type'], " ".join(test.iloc[x]['news'].split()[:10])))