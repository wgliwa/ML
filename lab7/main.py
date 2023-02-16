from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

bc = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size=0.25, random_state=12)

svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("Maszyny wektorow nosnych", accuracy_score(y_pred, y_test))

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("Jadro svm", accuracy_score(y_pred, y_test))

dtc = DecisionTreeClassifier(min_samples_leaf=5)
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
print("Drzewo decyzyjne", accuracy_score(y_pred, y_test))
