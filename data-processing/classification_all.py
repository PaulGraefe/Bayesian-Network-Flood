import numpy as np
import pandas as pd

def add_classification_to_csv(input_csv_path, column_name, low_threshold=None, high_threshold=None,
                              use_percentile=False):
    """
    Fügt einer CSV-Datei eine Klassifikationsspalte hinzu und speichert sie neu.

    Args:
        input_csv_path (str): Pfad zur Eingabe-CSV-Datei.
        column_name (str): Name der zu klassifizierenden Spalte.
        low_threshold (float): Schwellenwert für "Low" (nicht benötigt, wenn use_percentile=True).
        high_threshold (float): Schwellenwert für "High" (nicht benötigt, wenn use_percentile=True).
        use_percentile (bool): Ob Perzentile zur Klassifikation genutzt werden sollen.
    """
    data = pd.read_csv(input_csv_path)

    # Überprüfen, ob die Spalte existiert
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")

    # Konvertiere die Zielspalte zu numerischen Werten (nicht konvertierbare Werte werden NaN)
    data[column_name] = pd.to_numeric(data[column_name], errors='coerce')

    # Falls Perzentile genutzt werden, Schwellenwerte berechnen
    if use_percentile:
        low_threshold = data[column_name].quantile(0.33)
        high_threshold = data[column_name].quantile(0.67)

    # Überprüfen, ob Schwellenwerte gültig sind
    if low_threshold is None or high_threshold is None:
        raise ValueError("Thresholds must be defined or calculated.")

    # Klassifikationslogik
    def classify(value):
        if pd.isna(value):  # NaN-Werte behandeln
            return "NaN"
        elif value < low_threshold:
            return "Low"
        elif value > high_threshold:
            return "High"
        else:
            return "Medium"

    # Spalte hinzufügen
    data[f"{column_name}_class"] = data[column_name].apply(classify)

    # Speichere die aktualisierte CSV-Datei
    output_csv_path = input_csv_path.replace(".csv", "_classified.csv")
    data.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved with classification to: {output_csv_path}")
    return output_csv_path


# Spezifische Klassifikationsregeln für jede Datei
files_classification_rules = [


    ("/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/temperature.csv", "temperature", "fixed", 10, 21),
    ("/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/river_discharge.csv", "river_discharge",
     "fixed", 0.16, 0.5),
]
output_paths = []
for input_csv, column, low, high, use_percentile in files_classification_rules:
    try:
        output_path = add_classification_to_csv(input_csv, column, low, high, use_percentile)
        output_paths.append(output_path)
    except ValueError as e:
        print(f"Error processing {input_csv}: {e}")
