# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


# Wczytaj dane z pliku CSV
data = pd.read_csv('dane.csv')

# Przygotuj dane do uczenia
X = data[['kod_EAN', 'nazwa', 'dostawca']]
y = data['kategoria']

# Podziel dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Stwórz model drzewa decyzyjnego i wytrenuj go na danych treningowych
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Przewiduj kategorie dla danych testowych i oblicz dokładność modelu
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Dokładność modelu: {accuracy}')

# Przewiduj kategorie dla nowych danych
new_data = pd.read_csv('nowe_dane.csv')
X_new = new_data[['kod_EAN', 'nazwa', 'dostawca']]
y_new = clf.predict(X_new)

# Przypisz przewidziane kategorie do nowych danych i zapisz je w bazie danych
new_data['kategoria'] = y_new
new_data.to_csv('nowe_dane_z_kategoria.csv', index=False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
