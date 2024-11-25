import pandas as pd


def count_classifications(file_path, classification_column):
    """
    Zählt die Anzahl von Low, Medium und High Zuständen in einer CSV-Datei.

    Parameters:
    - file_path (str): Pfad zur CSV-Datei.
    - classification_column (str): Spaltenname, der die Klassifikation enthält.

    Returns:
    - dict: Ein Dictionary mit den Zählungen von "Low", "Medium", und "High".
    """
    try:
        # Datei einlesen
        data = pd.read_csv(file_path)

        # Sicherstellen, dass die Klassifikationsspalte vorhanden ist
        if classification_column not in data.columns:
            raise ValueError(f"Spalte '{classification_column}' nicht in der Datei gefunden.")

        # Zählungen durchführen
        counts = data[classification_column].value_counts().to_dict()

        # Ergebnisse zurückgeben
        return counts

    except Exception as e:
        print(f"Fehler beim Verarbeiten der Datei: {e}")
        return None


# Beispielaufruf
file_path = '/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/soil_moisture_classified.csv'
classification_counts = count_classifications(file_path, "soil_moisture_class")

if classification_counts:
    print("Zählungen der Klassifikationen:")
    for classification, count in classification_counts.items():
        print(f"{classification}: {count}")
