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

data_df = pd.read_csv('titanic.csv')

input_variables = ['Pclass', 'Sex', 'Fare', 'Embarked']
output_variable = ['Survived']

le = LabelEncoder()
data_df['Sex'] = le.fit_transform(data_df['Sex'])
data_df['Embarked'] = le.fit_transform(data_df['Embarked'])

X = data_df[input_variables]
y = data_df[output_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train.values.ravel())

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Točnost modela: {accuracy:.2f}')

# Vizualizacija podatkovnih primjera i granice odluke
h = .02  # korak mreže
x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, edgecolors='k')
plt.xlabel('Pclass')
plt.ylabel('Sex')
plt.title('Granica odluke za K-NN (K=5)')
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