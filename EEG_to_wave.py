import pandas as pd
from scipy.signal import welch
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
DANE="adhdata.csv" #tutaj wprowadzić ścieżkę pliku z odczytem EEG <-------- krok 1

raw_data = pd.read_csv(DANE) #wczytanie pliku z odczytem EEG

FS = 128
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
rows=[]

for person_id, g in raw_data.groupby('ID'): #przemiana danych z EEG na fale Beta, Theta oraz stosunek Theta/Beta
    row={'ID': person_id, 'Class': g['Class'].iloc[0]}
    for ch in CHANNELS:
        signal = g[ch].values
        frequency, psd = welch(signal, fs=FS, nperseg=FS * 2) #estymacja widma metodą Welcha

        theta_mask = (frequency >= 4) & (frequency <= 8) #filtr wyłapujący częstotliwośći 4-8Hz odpowiadające falom Theta
        beta_mask = (frequency >= 13) & (frequency <= 30) #filtr wyłapujący częstotliwośći 13-30Hz odpowiadające falom Beta

        theta = psd[theta_mask].sum() #suma przefiltrowanych wartości dla Theta
        beta = psd[beta_mask].sum() ##suma przefiltrowanych wartości dla Beta

        row[f'{ch}_theta'] = theta #wpisanie wyników Theta dla danego kanału
        row[f'{ch}_beta'] = beta #wpisanie wyników Beta dla danego kanału
        row[f'{ch}_ratio'] = theta/beta #wpisanie wyników stosunku dla danego kanału
    rows.append(row)
clean_data=pd.DataFrame(rows)
clean_data.to_csv("Clean_Data.csv", index=False) #eksport danych do zewnętrznego pliku CSV