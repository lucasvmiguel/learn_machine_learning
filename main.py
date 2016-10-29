# from sklearn import tree

# first video
# features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# labels = [0, 0, 1, 1]

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(features, labels)

# print clf.predict([[150, 0]])


#second video
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn import tree

# iris = load_iris()
# test_idx = [0, 50, 100]

# train_target = np.delete(iris.target, test_idx)
# train_data = np.delete(iris.data, test_idx, axis=0)

# test_target = iris.target[test_idx]
# test_data = iris.data[test_idx]

# clf = tree.DecisionTreeClassifier()
# clf.fit(train_data, train_target)

# #confirm that is alright
# print iris.target_names
# print iris.feature_names
# print iris.data[50]
# print test_target
# print clf.predict(test_data)

# #visualize tree decision(generate a pdf)
# import pydotplus 
# dot_data = tree.export_graphviz(clf, 
#     out_file=None,
#     feature_names=iris.feature_names,
#     class_names=iris.target_names,
#     filled=True, rounded=True, impurity=False) 
# graph = pydotplus.graph_from_dot_data(dot_data) 
# graph.write_pdf("iris.pdf") 


#video 4
# from sklearn import datasets

# iris = datasets.load_iris()

# X = iris.data
# y = iris.target

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# # from sklearn import tree
# # my_classifier = tree.DecisionTreeClassifier()
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()


# my_classifier.fit(X_train, y_train)

# predictions = my_classifier.predict(X_test)

# from sklearn.metrics import accuracy_score
# print accuracy_score(y_test, predictions)


#video 5
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()
# from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()


my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)