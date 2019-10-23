from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
import six
import pydot
import os
from subprocess import call

iris = load_iris()

df = pd.DataFrame({
    'sepal length' : iris.data[:,0],
     'sepal width' : iris.data[:,1],
      'petal length' : iris.data[:,2],
       'petal width' : iris.data[:,3],
       'species' : iris.target
})

X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]

y = df['species']

#podzial danych na train i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

#uczenie
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

y_predicted = classifier.predict(X_test)

#wyswietlanie
print(classification_report(y_test, y_predicted))

estimator = classifier.estimators_[5]
 
export_graphviz(estimator, out_file='tree.dot',
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False,
                precision = 2, filled = True)
 
# Convert to png using system command (requires Graphviz)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'], shell=True)
 
print(estimator)
 
 
dotfile = six.StringIO()
i_tree = 0
for tree_in_forest in classifier.estimators_:
    export_graphviz(tree_in_forest,out_file='tree.dot',
    feature_names=iris.feature_names,
    class_names = iris.target_names,
    filled=True,
    rounded=True)
    (graph,) = pydot.graph_from_dot_file('tree.dot')
    name = 'tree' + str(i_tree)
    graph.write_png(name+  '.png')
    os.system('dot -Tpng tree.dot -o tree.png')
    i_tree +=1