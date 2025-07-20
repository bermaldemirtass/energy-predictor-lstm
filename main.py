import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Veriyi oku
df = pd.read_csv("energydata_complete.csv", parse_dates=["date"], index_col="date")

# Saatlik ortalama hesapla
df_hourly = df["Appliances"].resample("H").mean().dropna()

# Normalize et
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_hourly.values.reshape(-1, 1))

# Sequence oluştur
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# Train-test böl
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Model kur
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# Eğit
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Tahmin
y_pred = model.predict(X_test)

# Normalize geri al
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Görselleştir
plt.figure(figsize=(12, 5))
plt.plot(y_test_rescaled[:100], label="Gerçek")
plt.plot(y_pred_rescaled[:100], label="Tahmin")
plt.legend()
plt.title("İlk 100 Saatlik Enerji Tüketimi Tahmini")
plt.xlabel("Saat")
plt.ylabel("Tüketim (Wh)")
plt.tight_layout()
plt.savefig("prediction_plot.png")
plt.show()

# Modeli kaydet
model.save("energy_predictor_lstm.h5")

