import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import joblib

def elm_fit(X, target, h, W=None, lambda_val=0.1):
    if W is None:
        W = np.random.uniform(-.6, .6, (h, len(X[0])))
    Hinit = X @ W.T
    # fungsi aktivasi Sigmoid
    H = 1 / (1 + np.exp(-Hinit))
    Ht = H.T
    
    # regularisasi ridge
    I = np.identity(h)
    Hp = np.linalg.inv(Ht @ H + lambda_val * I) @ Ht
    
    beta = Hp @ target

    return W, beta

def elm_predict(X, W, b, round_output=False):
    Hinit = X @ W.T
    # fungsi aktivasi Sigmoid
    H = 1 / (1 + np.exp(-Hinit))
    y = H @ b

    if round_output:
        y = [int(round(x)) for x in y]

    return y

# Memuat data dari file CSV menggunakan Pandas
DKI1 = pd.read_excel("D:\Data\Kuliah\TA\Projek-TA\DATA\Classification\DATA ISPU - classification.xlsx", sheet_name="DKI1")
DKI1['Kategori'] = DKI1['Kategori'].astype(float)
DKI1.info()

n_rows = 2192

# Mengambil kolom yang diperlukan
data = DKI1[:n_rows][['Tanggal', 'PM10', 'SO2', 'CO', 'O3', 'NO2', 'Kategori']]
# Melakukan normalisasi Min-Max pada fitur

# Memisahkan fitur (X) dan target (Y)
X = data[['PM10', 'SO2', 'CO', 'O3', 'NO2']].values
Y = data['Kategori'].values

# Memisahkan data menjadi training set dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

np.random.seed(0)

# Melakukan pelatihan model ELM
W_test, b_test = elm_fit(X_train, y_train, 10)

predict_test = elm_predict(X_test, W_test, b_test, round_output=True)

# Menyimpan model ELM sebagai file .sav
model = {'W': W_test, 'b': b_test}
joblib.dump(model, 'model_elm.sav')
