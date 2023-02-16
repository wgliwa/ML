from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
wine = datasets.load_wine()
standardizer = StandardScaler()
obs1 = [[1, 1, 1, 1], [0.75, 0.75, 0.75, 0.75]]
obs2 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]]
knni = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1).fit(standardizer.fit_transform(iris.data),
                                                                              iris.target)
print(knni.predict(obs1))
knnw = KNeighborsClassifier(n_neighbors=100, metric='minkowski', n_jobs=-1).fit(standardizer.fit_transform(wine.data),
                                                                                wine.target)
print(knnw.predict(obs2))

features_standardized = standardizer.fit_transform(iris.data)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])
search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 7, 8, 9, 10]}]

classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(features_standardized, iris.target)

print("best metric: ", classifier.best_estimator_.get_params()["knn__metric"])
print("best n: ", classifier.best_estimator_.get_params()["knn__n_neighbors"])
