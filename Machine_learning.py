import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data = pd.read_csv("Clean_Data.csv") #odczyt CSV z falami Beta, Theta i Ratio

SEED=42 #ustawienie folda/zmiana losowości w podziale <----------- krok 3
FOLD_COUNT=5 #ilość prób dla danego folda <---------- krok 3

X=data.iloc[:,2:] #rodzielenie danych
y=data.iloc[:,1]
groups=data['ID']

skgf= StratifiedGroupKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=SEED) #podział na grupy

def RFC():
    ACCURACY = []  # utworzenie list
    F1 = []
    y_test_list = []
    predicts_list = []
    for train_id, test_id in skgf.split(X, y, groups=groups):
        model = RandomForestClassifier(random_state=SEED) #utworzenie modelu RFC
        X_train, X_test = X.iloc[train_id], X.iloc[test_id]
        y_train, y_test = y.iloc[train_id], y.iloc[test_id]

        assert set(groups.iloc[train_id]).isdisjoint(set(groups.iloc[test_id])) #kontrola czy osoba nie trafiła do obu grup

        model.fit(X_train, y_train) #załączenie próbki danych szkoleniowych

        predicts = model.predict(X_test) #próba modelu

        y_test_list.extend(y_test)
        predicts_list.extend(predicts)

        ACCURACY.append(accuracy_score(y_test, predicts))
        F1.append(f1_score(y_test, predicts, pos_label='ADHD'))

    print("RFC-Średnia skuteczność: ", np.mean(ACCURACY)) #wypisanie średnich
    print("RFC-Średnia F1: ", np.mean(F1))
    print()
    print("RFC-Odchylenie standardowe skuteczności: ", np.std(ACCURACY)) #wypisanie odchylenia
    print("RFC-Odchylenie standardowe F1: ", np.std(F1))

    ConfusionMatrixDisplay.from_predictions(y_test_list, predicts_list) #Utworzenie matrycy RFC
    plt.show()
def LR():
    ACCURACY = []  # utworzenie list
    F1 = []
    y_test_list = []
    predicts_list = []
    for train_id, test_id in skgf.split(X, y, groups=groups):
        model = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000))]) #utworzenie modelu LR
        X_train, X_test = X.iloc[train_id], X.iloc[test_id]
        y_train, y_test = y.iloc[train_id], y.iloc[test_id]

        assert set(groups.iloc[train_id]).isdisjoint(set(groups.iloc[test_id])) #kontrola czy osoba nie trafiła do obu grup

        model.fit(X_train, y_train) #załączenie próbki danych szkoleniowych

        predicts = model.predict(X_test) #próba modelu

        y_test_list.extend(y_test)
        predicts_list.extend(predicts)

        ACCURACY.append(accuracy_score(y_test, predicts))
        F1.append(f1_score(y_test, predicts, pos_label='ADHD'))

    print("LR-Średnia skuteczność: ",np.mean(ACCURACY)) #wypisanie średnich
    print("LR-Średnia F1: ",np.mean(F1))
    print()
    print("LR-Odchylenie standardowe skuteczności: ", np.std(ACCURACY)) #wypisanie odchylenia
    print("LR-Odchylenie standardowe F1: " ,np.std(F1))

    ConfusionMatrixDisplay.from_predictions(y_test_list, predicts_list) #Utworzenie matrycy LR
    plt.show()
RFC() #wywolanie obu modeli
print()
LR()