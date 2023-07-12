import numpy as np
import joblib

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def elm_fit(X, target, h, W=None, lambda_val=1):
    if W is None:
        W = np.random.uniform(-.6, .6, (h, len(X[0])))
    Hinit = X @ W.T
    H = leaky_relu(Hinit)
    Ht = H.T
    
    # Menambahkan regularisasi ridge
    I = np.identity(h)
    Hp = np.linalg.inv(Ht @ H + lambda_val * I) @ Ht
    
    beta = Hp @ target
    y = H @ beta
    mape = sum(abs(y - target) / target) * 100 / len(target)
    return W, beta, mape


def elm_predict(X, W, b, round_output=False):
    Hinit = X @ W.T
    H = leaky_relu(Hinit)
    y = H @ b

    if round_output:
        y = [int(round(x)) for x in y]

    return y

import pandas as pd
from sklearn.preprocessing import minmax_scale

# Memuat data dari file CSV menggunakan Pandas
DKI1 = pd.read_excel("D:\Data\Kuliah\TA\Projek-TA\DATA\Classification\DATA ISPU - classification.xlsx", sheet_name="DKI1")
DKI1['Kategori'] = DKI1['Kategori'].astype(float)
DKI1.info()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

n_rows = 2192

# Mengambil kolom yang diperlukan
data = DKI1[:n_rows][['Tanggal', 'PM10', 'SO2', 'CO', 'O3', 'NO2', 'Kategori']]

# Memisahkan fitur (X) dan target (Y)
X = data[['PM10', 'SO2', 'CO', 'O3', 'NO2']].values
Y = data['Kategori'].values

# Memisahkan data menjadi training set dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

# Melakukan pelatihan model ELM
W_test, b_test, mape_test = elm_fit(X_train, y_train, 15)

np.random.seed(0)

# Menggunakan model ELM untuk melakukan prediksi
predict_test = elm_predict(X_test, W_test, b_test, round_output=True)

# Menyimpan model ELM sebagai file .sav
model = {'W': W_test, 'b': b_test}
joblib.dump(model, 'model_leaky.sav')