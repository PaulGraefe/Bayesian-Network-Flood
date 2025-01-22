import pandas as pd

# Die ursprüngliche Datei einlesen (mit Tabulator als Trennzeichen, \t)
input_file = '/bayesian_network/InferenceData/mbVali.csv'  # Pfad zur Datei
data = pd.read_csv(input_file, sep='\t')  # Tabulator-getrennte Datei einlesen

# Die Datei mit ; als Trennzeichen speichern
output_file = '/bayesian_network/InferenceData/mbValiOutput.csv'  # Ziel-Datei
data.to_csv(output_file, sep=',', index=False)

print(f"Die Datei wurde erfolgreich als '{output_file}' mit ';' als Trennzeichen gespeichert.")
