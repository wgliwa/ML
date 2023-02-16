from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def crossval():
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    print("modelu sprawdzianu krzyzowego, srednia: ",
          cross_val_score(pipeline, x, y, cv=kf, scoring="accuracy", n_jobs=-1).mean())


def linear():
    ols = LinearRegression()
    ols.fit(x_train, y_train)
    print("modelu regresji bazowej: ", ols.score(x_test, y_test))


def forest():
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    print("modelu klasyfikacji bazowej: ", classifier.score(x_test, y_test))


def binary_pred():
    logit = LogisticRegression()
    y_hat = logit.fit(x_train, y_train).predict(x_test)
    print("prognozy klasyfikatora binarnego: ", accuracy_score(y_test, y_hat))


def binary_thresh():
    logit = LogisticRegression()
    logit.fit(x_train, y_train)
    print("progowania klasyfikatora binarnego: ", roc_auc_score(y_test, logit.predict_proba(x_test)[:, 1]))


x, y = datasets.make_classification(10000, 3, n_informative=3, n_redundant=0, n_classes=2, random_state=3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3)

print("Ocena:")
crossval()
linear()
forest()
binary_pred()
binary_thresh()
