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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from mlxtend.plotting import plot_decision_regions

##################################################
#1. zadatak
##################################################

#učitavanje dataseta
data = pd.read_csv('titanic.csv')

#a)
female = data[data['Sex'] == 'female']
print(f"Broj žena: {len(female)}")

#b)
not_survived = data[data['Survived'] == 0]
print(f"Broj osoba koje nisu prezivjele {len(not_survived)}")

percentage_non_survived = (len(not_survived) / len(data)) * 100
print(f"Postotak osoba koje nisu preživjele: {percentage_non_survived} %")

#c)
survived_male = data[(data['Survived'] == 1) & (data['Sex'] == 'male')]
survived_female = data[(data['Survived'] == 1) & (data['Sex'] == 'female')]

total_survived = len(survived_female) + len(survived_male)

male_survived_percentage = (len(survived_male) / total_survived) * 100
female_survived_percentage = (len(survived_female) / total_survived) * 100

print(f"Preživjelih muškaraca: {male_survived_percentage} %")
print(f"Preživjelih žena: {female_survived_percentage} %")

genders = ['male', 'female']
colors = ['green', 'yellow']
percentage = [male_survived_percentage, female_survived_percentage]

plt.figure()
plt.bar(genders, percentage, color=colors)
plt.ylabel("Postoci")
plt.title("Postotak preživjelih po poslu")
plt.show()

#d)
survived_males_average_age = survived_male['Age'].mean()
survived_females_average_age = survived_female['Age'].mean()

print(f"Prosjecna dob prezivjelog muskarca: {survived_males_average_age}")
print(f"Prosjecna dob prezivjele zene: {survived_females_average_age}")

#e)
oldest_survived_male = survived_male.groupby('Pclass')['Age'].max()

print("Najstariji prezivjeli muskarac u svakoj klasi")
print(oldest_survived_male)

##################################################
#2. zadatak
##################################################

#učitavanje dataseta
data_df = pd.read_csv('titanic.csv')

input_variables = ['Pclass', 'Sex', 'Fare', 'Embarked']
output_variable = ['Survived']

data_df = data_df.dropna(axis=0)
data_df.drop_duplicates()
data_df = data_df.reset_index(drop=True)

X = data_df[input_variables]
y = data_df[output_variable]

titanic = pd.get_dummies(X, columns=['Sex', 'Embarked'], dtype=float)
print("OIDHGOSDIFHSDOFUHSDOFUSDOUHROFUHW")
print(titanic)

X_train, X_test, y_train, y_test = train_test_split(titanic, y, test_size=0.3, random_state=42)

#a)
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train, y_train)
y_train_p_KNN = KNN_model.predict(X_train)
y_test_p_KNN = KNN_model.predict(X_test)

plot_decision_regions(X_train, y_train, clf=KNN_model, legend=2)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

#b)
print("KNN: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

#c)
KNN_model = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1, 100)}
KNN_gscv = GridSearchCV(KNN_model, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
KNN_gscv.fit(X_train, y_train)
k = KNN_gscv.best_params_['n_neighbors']
print(k)

#d)
KNN_model.set_params(n_neighbors=k)
KNN_model.fit(X_train, y_train)
y_train_p_KNN = KNN_model.predict(X_train)
y_test_p_KNN = KNN_model.predict(X_test)

print(f"KNN K: {k}")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

#plot_decision_regions(X_train, y_train, classifier=KNN_model)
plot_decision_regions(X_train, y_train, clf=KNN_model, legend=2)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

'''
##################################################
#3. zadatak
##################################################

#učitavanje podataka:
data_df = pd.read_csv('titanic.csv')

input_variables = ['Pclass', 'Sex', 'Fare', 'Embarked']
output_variable = ['Survived']

data_df = data_df.dropna(axis=0)
data_df.drop_duplicates()
data_df = data_df.reset_index(drop=True)

X = data_df[input_variables]
y = data_df[output_variable]

titanic = pd.get_dummies(X, columns=['Sex', 'Embarked'], dtype=float)
print("IUFHGDOSUIHFOSUDHFSDOUFHOSDUFHSDOIFHSDOFIHDSOI")
print(titanic)

X_train, X_test, y_train, y_test = train_test_split(titanic, y, test_size=0.25, random_state=42)

#a)
model = keras.Sequential()
model.add(layers.Input(shape=(4,)))
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(4, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

#b)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', ])

#c)
epochs = 100
batch_size = 5
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#d)
model.save('FCN/zad_model.h5')

#e)
model = keras.models.load_model('FCN/zad_model.h5')
score = model.evaluate(X_test, y_test, verbose=0)
print(score)

#f)
predictions = np.around(model.predict(X_test)).astype(np.int32)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions))
disp.plot()
plt.show()
'''