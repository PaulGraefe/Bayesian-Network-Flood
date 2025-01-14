import pandas as pd
import matplotlib.pyplot as plt

# Pfad zur CSV-Datei (ändere diesen Pfad, falls notwendig)

# CSV einlesen
# Verwende den Semikolon (;) als Trennzeichen, da die Datei so strukturiert ist
data = pd.read_csv("/Users/paulgraefe/PycharmProjects/scientificProject/bayesian_network/InterferenceData/mbValiOutput.csv", sep=",")

# Die ersten Zeilen anzeigen, um sicherzustellen, dass die Daten korrekt eingelesen wurden
print(data.head())

# Spaltennamen überprüfen
print("Spaltennamen:", data.columns)

# Die Spalte mit den Wahrscheinlichkeiten (z.B. 'FLOOD_RISK_Yes_Probability') analysieren
probabilities = data['FLOOD']

# Statistiken berechnen
print("Statistiken:")
print("Anzahl:", probabilities.count())
print("Durchschnitt:", probabilities.mean())
print("Median:", probabilities.median())
print("Standardabweichung:", probabilities.std())
print("Minimum:", probabilities.min())
print("Maximum:", probabilities.max())

# Histogramm der Wahrscheinlichkeiten plotten
plt.hist(probabilities, bins=20, color='skyblue', edgecolor='black')
plt.title("Verteilung der Flutrisikowahrscheinlichkeiten für das Gebiet Miedelsbach")
plt.xlabel("Wahrscheinlichkeit")
plt.ylabel("Anzahl")
plt.show()
