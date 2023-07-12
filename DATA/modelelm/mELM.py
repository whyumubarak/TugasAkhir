import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

def elm_fit(X, target, h, W=None, lambda_val=0.01):
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

# Memuat data dari file CSV menggunakan Pandas
DKI1 = pd.read_excel("D:\Data\Kuliah\TA\Projek-TA\DATA\Classification\DATA ISPU - classification.xlsx", sheet_name="DKI1")
DKI1['Kategori'] = DKI1['Kategori'].astype(float)

n_rows = 2192

# Mengambil kolom yang diperlukan
data = DKI1[:n_rows][['Tanggal', 'PM10', 'SO2', 'CO', 'O3', 'NO2', 'Kategori']]
#data = DKI1[['Tanggal', 'PM10', 'SO2', 'CO', 'O3', 'NO2', 'Kategori']]
# Melakukan normalisasi Min-Max pada fitur
data[['PM10', 'SO2', 'CO', 'O3', 'NO2']] = minmax_scale(data[['PM10', 'SO2', 'CO', 'O3', 'NO2']])

# Memisahkan fitur (X) dan target (Y)
X = data[['PM10', 'SO2', 'CO', 'O3', 'NO2']].values
Y = data['Kategori'].values

# Memisahkan data menjadi training set dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

# Melakukan pelatihan model ELM
new_W = np.array([[-0.2067117, 0.32006617, 0.1458637, -0.17858387, -0.012993],
                  [0.1555636, -0.15292709, 0.55353617, 0.27372434, 0.31595392],
                  [-0.33564727, 0.01557412, 0.33728987, 0.45583973, -0.39229665],
                  [-0.03250283, 0.47895379, 0.25719228, -0.33582577, -0.2595521],
                  [0.55724535, 0.02011247, -0.57450425, 0.50479632, -0.43732137],
                  [0.0917777, -0.51987686, -0.38332002, -0.47096607, -0.30595839],
                  [0.02658457, 0.14434701, -0.15484402, -0.23192934, 0.2290791],
                  [0.56009997, -0.0028923, -0.28107938, 0.44556186, 0.18976659],
                  [0.13485597, -0.07721839, 0.12703944, 0.28319432, 0.40710455],
                  [0.32655645, 0.56435527, -0.47712076, -0.34197189, 0.04897925],
                  [0.12109037, 0.09006689, 0.59329199, 0.04987426, 0.07884046],
                  [-0.55012228, -0.0186138, -0.25608019, -0.34421418, 0.37364695],
                  [0.20248293, -0.35570922, 0.11776813, 0.32418357, 0.00589534],
                  [-0.10853963, -0.12406312, 0.38784341, -0.14688425, 0.48300669],
                  [-0.00341443, 0.38429398, 0.13777226, -0.07559466, -0.17076889]])

new_b = np.array([0.49786702, 2.2153489, 3.64313601, 1.63783446, -1.13444277,
                  0.26943279, 2.6449985, 6.06646338, -4.37808228, -0.01232608,
                  0.677152, -0.05372555, -1.16944742, -0.46151583, -2.22764204])

W_test, b_test = new_W, new_b

# Simpan model ELM ke dalam file sav
joblib.dump((W_test, b_test), 'model_elm.sav')