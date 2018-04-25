import data_sort

from sklearn.ensemble import AdaBoostClassifier

boost = AdaBoostClassifier(algorithm = "SAMME", n_estimators = 100)

number_of_features = 15
training_data = data_sort.makeSet("adult_data_big.txt", number_of_features)
test_data = data_sort.makeSet("adult_data_test.txt", number_of_features)

X_train, y_train = data_sort.split(training_data)
X_test, y_test = data_sort.split(test_data)

boost.fit(X_train, y_train)
print(boost.score(X_test, y_test))