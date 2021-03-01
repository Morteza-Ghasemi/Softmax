from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.classifier import SoftmaxRegression
from sklearn import datasets

iris = datasets.load_iris()
#X = iris.data[:, [2, 3]]
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


############# softmax Regresion ###############

# Fitting softmax regression to the tranning set
foft_regressor = SoftmaxRegression()
foft_regressor.fit(X_train, y_train)

# Predicting the test set result
y_pred = foft_regressor.predict(X_test)

print("############ softmax Regression ############")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

