import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras import layers
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, \
    precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
from sklearn.cluster import KMeans

#učitavanje dataseta
iris = datasets.load_iris()

##################################################
#1. zadatak
##################################################

print("Nazivi stupaca:")
print(iris.feature_names)
print(iris.target_names)
print("=======================================================")

#a)
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

data['target'] = iris.target

Setosa = data[(data['target'] == 0)]
Varsicolor = data[(data['target'] == 1)]
Virginica = data[(data['target'] == 2)]

plt.scatter(Virginica['sepal length (cm)'], Virginica['petal length (cm)'], c='green', label='Virginica')
plt.scatter(Setosa['sepal length (cm)'], Setosa['petal length (cm)'], c='gray', label='Setosa')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.title('Usporedba Virginica i Setosa')
plt.legend(loc='upper left')
plt.show()

#b)
virginica_max_width = Virginica['sepal width (cm)'].max()
setosa_max_width = Setosa['sepal width (cm)'].max()
varsicolor_max_width = Varsicolor['sepal width (cm)'].max()

plt.bar(['Virginica', 'Varsicolor', 'Setosa'], [virginica_max_width, setosa_max_width, varsicolor_max_width])
plt.xlabel('Klasa cvijeta')
plt.ylabel('Najveća širina čašice (cm)')
plt.title('Najveća širina čašice po klasi cvijeta')
plt.xticks(rotation=90)
plt.show()

#c)
mean_sepal_width_setosa = Setosa['sepal width (cm)'].mean()

setosa_bigger_than_average = Setosa[(Setosa['sepal width (cm)'] > mean_sepal_width_setosa)].count()

print("Broj jedinki klase Setosa s većom širinom čašice od prosječne vrijednosti:", setosa_bigger_than_average)

##################################################
#2. zadatak
##################################################

print(data.shape)

data = data.drop_duplicates()
data = data.dropna(axis=0)
data.reset_index()

print(data.shape)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#a)
J_K_array = []
for i in range(1, 10):
    km = KMeans(n_clusters=i, init='k-means++', n_init=5, random_state=0)
    km.fit(X)
    J_K_array.append(km.inertia_)

#b)
plt.figure()
plt.plot(range(1, 10), J_K_array, marker=".")
plt.title('Lakat metoda')
plt.xlabel('K vrijednosti')
plt.ylabel('J vrijednosti')
plt.tight_layout()

#c)
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

km = KMeans(n_clusters=3, init='k-means++', n_init=5, random_state=0)
km.fit(X)
labels = km.fit_predict(X)

#d)
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s=100, c='yellow', label='Setosa')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s=100, c='orange', label='Versicolour')
plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s=100, c='green', label='Virginica')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.title('K-means Clustering Iris Dataset')
plt.legend()
plt.show()

print('accuracy k-means')

#e)
counter = 0
for i in range(0, len(labels)):
    if labels[i] == y[i]:
        counter = counter + 1

print(f'{counter / len(labels) * 100} %')

##################################################
#3. zadatak
##################################################

iris = datasets.load_iris()

y = y.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
#encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

#a)
model = keras.Sequential()
model.add(layers.Flatten(input_shape=(4,)))
model.add(layers.Dense(12, activation="relu"))  #prvi
model.add(layers.Dropout(0.2))
model.add(layers.Dense(7, activation="relu"))  #drugi
model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation="relu"))  #treci
model.add(layers.Dense(3, activation="softmax"))

model.summary()

#b)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", ])

#c)
history = model.fit(X_train_n, y_train, batch_size=7, epochs=450, validation_split=0.1)

#d)
model.save('FCN/zad_model.h5')

#e)
model = keras.models.load_model('FCN/zad_model.h5')

test_loss, test_acc = model.evaluate(X_test_n, y_test, verbose=0)

print(test_acc)
print(test_loss)

#f)
y_test_p = model.predict(X_test_n)
y_label = np.argmax(y_test, axis=1)
predict_label = np.argmax(y_test_p, axis=1)
length = len(y_test_p)

cm = confusion_matrix(y_label, predict_label)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

print(classification_report(y_label, predict_label))

accuracy = np.sum(y_label == predict_label)/length * 100
print("Accuracy of the dataset", accuracy)
