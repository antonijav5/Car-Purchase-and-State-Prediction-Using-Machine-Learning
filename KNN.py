import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import KNNClass as kn


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
data = pd.read_csv('car_state.csv')
# Ispis prvih 5 redova DataFrame-a
print("Prvih pet redova:")
print(data.head())
print("\n")
print("Poslednjih pet redova:")
print(data.tail())
print("\n")
print("Informacije o tabeli")
print(data.info())
print("\n")
print("Generisanje opisne statistike:")
print(data.describe(percentiles=[0.2, 0.5, 0.8], include=['object', 'float', 'int']))
print("\n")
# zavisnost statusa od cene
# pogodniji je ovaj tip grafika
sb.displot(data, x='buying_price', hue='status', multiple='fill')
plt.show()
# zavisnost statusa od odrzavanja
# pogodniji je ovaj tip grafika
sb.displot(data, x='maintenance', hue='status', multiple='fill')
plt.show()
# zavisnost statusa od broja vrata
# pogodniji je ovaj tip grafika
sb.displot(data, x='doors', hue='status', multiple='fill')
plt.show()
# zavisnost statusa od broja sedista
# pogodniji je ovaj tip grafika
sb.displot(data, x='seats', hue='status', multiple='fill')
plt.show()
# zavisnost statusa od velicine gepeka
# pogodniji je ovaj tip grafika
sb.displot(data, x='trunk_size', hue='status', multiple='fill')
plt.show()
# zavisnost statusa od bezbednosti
# pogodniji je ovaj tip grafika
sb.displot(data, x='safety', hue='status', multiple='fill')
plt.show()

# izbor atributa za treniranje modela-svi su izabrani, jer nijedan nije nezavisan


# ipak transformisemo u BROJEVE zbog racunanja
data['buying_price'].replace(to_replace=['low', 'medium', 'high', 'very high'], value=[1, 2, 3, 4], inplace=True)
data['maintenance'].replace(to_replace=['low', 'medium', 'high', 'very high'], value=[1, 2, 3, 4], inplace=True)
data['doors'].replace(to_replace=['2', '3', '4', '5 or more'], value=[1, 2, 3, 4], inplace=True)
data['seats'].replace(to_replace=['2', '4', '5 or more'], value=[1, 2, 3], inplace=True)
data['trunk_size'].replace(to_replace=['small', 'medium', 'big'], value=[1, 2, 3], inplace=True)
data['safety'].replace(to_replace=['low', 'medium', 'high'], value=[1, 2, 3], inplace=True)

n = data.shape[0]
k = np.sqrt(n).astype(int)

# trazi se fja greske i preciznost modela-KNeighborsClassifier class i moje
x = np.array(data.iloc[:, 0:6])  # end index is exclusive
y = np.array(data['status'])
error = []  # njihov error
error0 = []  # moj error
x_train, \
x_test, \
y_train, \
y_test = train_test_split(x, y, test_size=0.33, random_state=42)
myKNN = kn.KNNClass()
# greske za izbor razlicitih k (jedina vrsta greske koju sam nasla)
# moja klasa
arr = [5, 15, 37, 50]
sum = 0
for i in arr:
    knn0 = kn.KNNClass()
    # knn0.fit(x_train,y_train)
    pred_i = knn0.k_nearest_neighbor(x_train, y_train, x_test, i)
    error0.append(np.mean(pred_i != y_test))
    sum += np.mean(pred_i != y_test)
print("The error of out algorithm is approximately {:0.2f}%".format(sum/4))
plt.figure(figsize=(12, 6))
plt.plot([5, 15, 37, 50], error0, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value - KNNClass ')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

# njihova klasa
arr = [5, 15, 37, 50]
sum = 0
for i in arr:
    knn1 = KNeighborsClassifier(n_neighbors=i)

    knn1.fit(x_train, y_train)

    pred_i = knn1.predict(x_test)
    error.append(np.mean(pred_i != y_test))
    sum += np.mean(pred_i != y_test)

print("The error of their algorithm is approximately {:0.2f}%".format(sum/4))
plt.figure(figsize=(12, 6))
plt.plot([5, 15, 37, 50], error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value - KNeighborsClassifier')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
# preciznost
# moja KNN klasa
predictions = myKNN.k_nearest_neighbor(x_train, y_train, x_test, k)
# accuracy mog modela
accuracy = kn.accuracy_score(y_test, predictions)
print("The accuracy of our classifier is {:0.2f}%".format(100*accuracy))

# njihova klasa
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
their_score = knn.score(x_test, y_test)*100.00
print("The accuracy of their classifier is {:0.2f}%".format(their_score))
