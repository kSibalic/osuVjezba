'''
ZADATAK 0.0.1
'''
import keras
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import pandas as pd
import math

from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, \
    classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

# a)
data = np.loadtxt('pima-indians-diabetes.csv', delimiter=",", dtype=float, skiprows=1)
data = data[1::]
data = np.array(data, np.float64)
print(f"Broj izmjerenih ljudi: {len(data)}")

# b)
data = pd.read_csv('pima-indians-diabetes.csv')
print(f'\nIzostale vrijednosti: {data.isnull().sum()}')
print(f'Duplicirane vrijednosti: {data.duplicated().sum()}')

# c)
plt.scatter(data['BMI'], data['Age'], color='blue', alpha=0.8)
plt.xlabel('BMI')
plt.ylabel('Age')
plt.title('BMI vs Age')
plt.show()

# d)
print(f'\nSrednja vrijednost: {data['BMI'].mean()}')
print(f'Min vrijednost: {data['BMI'].min()}')
print(f'Max vrijednost: {data['BMI'].max()}')

# e)
selectedDiabetes = data[(data['Outcome'] == 1)]
selectedNonDiabetes = data[(data['Outcome'] == 0)]

print(f'\nSrednja vrijednost (D): {selectedDiabetes['BMI'].mean()}')
print(f'Min vrijednost (D): {selectedDiabetes['BMI'].min()}')
print(f'Max vrijednost (D): {selectedDiabetes['BMI'].max()}')

print(f'\nSrednja vrijednost (ND): {selectedNonDiabetes['BMI'].mean()}')
print(f'Min vrijednost (ND): {selectedNonDiabetes['BMI'].min()}')
print(f'Max vrijednost (ND): {selectedNonDiabetes['BMI'].max()}')

print(f'\nBroj ljudi s dijabetesom: {len(selectedDiabetes)}')

'''
ZADATAK 0.0.2
'''
data = np.loadtxt('pima-indians-diabetes.csv', delimiter=",", dtype=float, skiprows=1)

# a)
X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
linearModel = lm.LogisticRegression(max_iter=1000)
linearModel.fit(X_train, y_train)
#print(linearModel.coef_)

# b)
y_pred = linearModel.predict(X_test)
print()
print(classification_report(y_test, y_pred))

# c)
cm = confusion_matrix(y_test, y_pred)
print("\nMatrica zabune: ", cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()

# d)
print(f'\nTočnost: {accuracy_score(y_test, y_pred)}')
print(f'Preciznost: {precision_score(y_test, y_pred)}')
print(f'Odziv: {recall_score(y_test, y_pred)}')
print()

'''
ZADATAK 0.0.3
'''
# a)
model = keras.Sequential()
model.add(layers.Input(shape=(8,)))
model.add(layers.Flatten())
model.add(layers.Dense(12, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

# b)
print()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# c)
batch_size = 10
epochs = 150
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
print()

'''
# d)
model.save('FCN/zad_model.h5')

# e)
model = keras.models.load_model('FCN/zad_model.h5')

evaluation = model.evaluate(X_test, y_test)

print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])

# f)
model = keras.models.load_model('FCN/zad_model.h5')

predictions = model.predict(X_test)

predicted_classes = np.argmax(predictions)
true_classes = np.argmax(y_test)
conf_matrix = confusion_matrix(true_classes, predicted_classes)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
'''