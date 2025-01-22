import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# CSV-Datei einlesen
file_path = "/bayesian_network/InferenceData/validierung_r_v3.csv"  # Pfad zur CSV-Datei
data = pd.read_csv(file_path)

# Extrahiere tatsächliche Werte und Modellvorhersagen
y_true = data['actualValue']  # Tatsächliche Werte
y_pred = data['FLOOD']  # Modellvorhersagen

# Residuen berechnen
residuen = y_true - y_pred

# MSE und RMSE berechnen
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("MSE: " + str(mse))
print("RMSE: " + str(rmse))

# Residuen grafisch darstellen
plt.scatter(y_pred, residuen, label="Residuen")
plt.xlabel("Vorhergesagte Werte (Flutrisiko Wahrscheinlichkeit)")
plt.ylabel("Residuen")
plt.title("Residuenanalyse")

# Setze die y-Achse so, dass sie bei 0 beginnt
plt.ylim(bottom=0)

# RMSE-Wert unter die Legende setzen
plt.gcf().text(0.72, 0.7, f"RMSE: {rmse:.3f}", fontsize=12, color="blue", bbox=dict(facecolor='white', alpha=0.5))

# Legende hinzufügen
plt.legend()

# Plot anzeigen
plt.show()
