'''
ZADATAK 0.0.1
'''
import keras
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import pandas as pd

from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

'''
diabetes = pd.read_csv('pima-indians-diabetes.csv')
data = np.array(diabetes)
data.shape[0]

b)
np.set_printoptions(suppress=True, precision=2)
unique = np.unique(data, axis=0)
unique = unique[unique[:,5] != 0]
unique = unique[unique[:,7] != 0]

c)
plt.plot()
plt.scatter(unique[:,5], unique[:,7], color="black")
plt.xlabel("BMI")
plt.ylabel("age")
plt.title("pito")
plt.show()

d)
print(unique[:,5].min(), unique[:,5].argmin())
print(unique[:,5].max(), unique[:,5].argmax())
'''

# a)
print("=====================================================")
print("\nZadatak 0.0.1 pod a)")
data = np.loadtxt('pima-indians-diabetes.csv', delimiter=",", dtype="str", skiprows=9)
data = np.array(data, np.float64)
print(f"\nBroj izmjerenih ljudi: {len(data)}")

# b)
print("=====================================================")
print("\nZadatak 0.0.1 pod b)")
df = pd.DataFrame(data)
print(f'\nBroj izlostalih vrijednosti: {df.isnull().sum()}')
print(f'Broj dupliciranih vrijednosti: {df.duplicated().sum()}')

df = df.drop_duplicates()
df = df.dropna(axis=0)                          # treba izbacit sve 0 iz BMI

data = data[data[:, 5] != 0.0]                  # izbacivanje sve s 0.0 BMI
data = data[data[:, 7] != 0.0]                  # izbacivanje sve s 0.0 BMI
df = pd.DataFrame(data)                         # ponovno kreiranje df ali ovaj put s ociscenim podacima bez redaka s BMI 0.0
print(f'Broj preostalih: {len(df)}')

# c)
print("=====================================================")
print("\nZadatak 0.0.1 pod c)")
plt.figure()
plt.scatter(x=data[:, 7], y=data[:, 5], color='blue', alpha=0.7)
plt.xlabel('BMI')
plt.ylabel('Age')
plt.title('BMI vs Age')
plt.show()

# d)
print("=====================================================")
print("\nZadatak 0.0.1 pod d)")
print(f'Srednja vrijednost: {df[5].mean()}')
print(f'Min vrijednost: {df[5].min()}')
print(f'Max vrijednost: {df[5].max()}')

# e)
print("=====================================================")
print("\nZadatak 0.0.1 pod e)")
selectedDiabetes = df[(df[8] == 1)]
selectedNonDiabetes = df[(df[8] == 0)]

print(f'\nBroj ljudi s dijabetesom: {len(selectedDiabetes)}')

print(f'\nSrednja vrijednost (D): {selectedDiabetes[5].mean()}')
print(f'Min vrijednost (D): {selectedDiabetes[5].min()}')
print(f'Max vrijednost (D): {selectedDiabetes[5].max()}')

print(f'\nSrednja vrijednost (ND): {selectedNonDiabetes[5].mean()}')
print(f'Min vrijednost (ND): {selectedNonDiabetes[5].min()}')
print(f'Max vrijednost (ND): {selectedNonDiabetes[5].max()}')

'''
ZADATAK 0.0.2
'''
data = np.loadtxt('pima-indians-diabetes.csv', delimiter=",", dtype=np.float64, skiprows=9)

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

# d)
model.save('FCN/zad_model.h5')

# e)
model = keras.models.load_model('FCN/zad_model.h5')

evaluation = model.evaluate(X_test, y_test)

print("Loss:", evaluation[0])
print("Accuracy:", evaluation[1])

# f)
y_pred = model.predict(X_test)
y_pred = np.around(y_pred).astype(np.int32)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
plt.show()