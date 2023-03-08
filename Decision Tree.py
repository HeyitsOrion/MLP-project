# Decision Tree
import sys
import matplotlib
matplotlib.use('Agg')

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

y = y_train['imdb_rating']
y = y.tolist()
threshold_1=np.percentile(y_train,33)
threshold_2=np.percentile(y_train,66)

y_category=[]
for i in range(0,len(y)):
    if y[i] <= threshold_1:
        y_category.append("Low")
    elif y[i] >= threshold_2:
        y_category.append("High")
    else:
        y_category.append("Medium")
print(y)
print(y_category)

X = x_train.drop(['n_words'],axis=1)
features = ['season','episode','total_votes','n_lines','n_directions','n_speak_char']

dtree = DecisionTreeClassifier(criterion = "gini", splitter = 'random', max_leaf_nodes = 10, min_samples_leaf = 5, max_depth= 5)
dtree = dtree.fit(X,y_category)

#tree.plot_tree(dtree,feature_names=features)
#plt.savefig(sys.stdout.buffer)
#sys.stdout.flush()

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtree,feature_names=features,class_names=["Low","Medium","High"],filled=True)
fig.savefig("decistion_tree.png")

<img src="./decistion_tree.png",width=320,heigth=240>

y = y_test
threshold_1=np.percentile(y,33)
threshold_2=np.percentile(y,66)

ytest=[]
for i in range(0,len(y)):
    if y[i] <= threshold_1:
        ytest.append("Low")
    elif y[i] >= threshold_2:
        ytest.append("High")
    else:
        ytest.append("Medium")
xtest=x_test.drop(['n_words'],axis=1)

# Calculate the correct classification score
score = dtree.score(xtest, ytest)
score