## OGÓLNE
Projekt klasyfikatora ADHD na podstawie odczytów EEG (121 osób, 19 kanałów) poprzez uczenie maszynowe (Random Forest Classifier i Logistic Regression)

## DANE
Dane pochodzą z Kaggle (https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd?resource=download), plik .CSV ważący 267 MB zawiera w sobie:

1. ID - Numer identyfikacyjny przypisany do danej osoby od której pochodzi odczyt

2. Class - Określa przynależność do grup badawczych (ADHD = grupa z stwierdzonym ADHD, Control = grupa kontrolna)

3. 19 kanałów - każdy odczyt z innej części mózgu, na ich podstawie będzie oceniane czy osoba ma ADHD

## CEL
Głównym zamiarem jest utworzenie modelu, który na podstawie odczytu EEG przekształcony metodą Welcha (podczas skryptu EEG_to_wave.py) przemienionego na fale Theta, Beta i stosunek Theta/Beta (wyodrębnienie pasm 4-8Hz dla Theta i 13-30Hz dla Beta) bedzie w stanie ocenić czy osoba ma ADHD. Funkcje StratifiedGroupKFold i assert zapewniają, że żadna z osób nie trafi zarówno do grupy treningowej i do grupy sprawdzającej. Użyte modele Random Forest Classifier i Logistic Regression korzystające z oczyszczonych danych przewidują czy dana osoba ma ADHD, dodatkowo regresja logistyczna ma skaler, który zapewnia stabilne i poprawne ocenianie ważności poszczególnych wartości.

## WYNIKI
Wyniki są przedstawiane w postaci średnich oraz odchylenia skuteczności i F1 z poszczególnych modelów, następnie zostaje ukazana forma graficzna poprzez confusion matrix. Przy załączonych danych model RFC wypada lepiej (ACCURACY 0,636 ± 0,096; F1 0,652 ± 0,090) od LR (ACCURACY 0,570 ± 0,055; F1 0,578 ± 0,064)

## OGRANICZENIA
Model ma małą próbkę oraz korzysta tylko z jednego wiersza na osobę, modele wystawiają predykcję tylko na podstawie 3 parametrów dla każdego kanału. Program jest bardzo wstępny, jest dużo miejsca do rozwoju.

## INSTRUKCJA
1.W EEG_to_wave.py należy podpiąc plik z odczytem EEG, aby to zrobić można:
a) umieścić plik .csv w tym samym folderze co program, oraz zmień nazwę dla stałej DANE: DANE="{nazwa_pliku}.csv" np. DANE="adhdata.csv"
b) umieścić ścieżkę pliku dla stałej DANE: DANE="{ścieżka_do_pliku}" np. DANE="/home/{nazwa_użytkownika}/PycharmProjects/WelcomeScreen/adhdata.csv"

2.Uruchomić program EEG_to_wave.py (upewnij się, że Clean_Data.csv jest w tym samym folderze co skrypt)

3.Zmienić SEED i FOLD_COUNT na wybrane przez siebie (14 i 15 linijka w pliku Machine_learning.py)

4.Uruchomić program Machine_learning.py

## REQUREMENTS
pip install -r requirements.txt
