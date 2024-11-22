import pandas as pd



def classify_category(data, column='temperature', low_threshold=10.0, high_treshold=21.0):
    """
    Klassifiziert die tägliche Regenmenge in High, Medium und Low basierend auf den Schwellenwerten.

    :param data: DataFrame mit einer Spalte für Regenmenge
    :param column: Name der Spalte mit den täglichen Regenwerten
    :param low_threshold: Obergrenze für die Klassifikation 'Low'
    :param high_treshold: Obergrenze für die Klassifikation 'Medium'
    :return: DataFrame mit einer zusätzlichen Spalte 'rainfall_class'
    """

    def classify(rainfall):
        if rainfall > high_treshold:
            return "High"
        elif rainfall > low_threshold:
            return "Medium"
        else:
            return "Low"

    # Anwenden der Klassifikationslogik
    if column not in data.columns:
        raise ValueError(f"Die Spalte '{column}' wurde im DataFrame nicht gefunden.")

    data['temperature_class'] = data[column].apply(classify)
    return data


# Datei laden
data = pd.read_csv(
    '/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/temperature.csv')

# Klassifikation anwenden
classified_rainfall_data = classify_category(data, column='temperature')

classified_rainfall_data.to_csv('/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/temperature_classified.csv')

classified_rainfall_data.head()  # Zeigt die ersten Zeilen zur Überprüfung