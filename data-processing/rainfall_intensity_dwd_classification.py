import pandas as pd

def classify_rainfall_daily(file_path):
    # Datei einlesen
    data = pd.read_csv(file_path, delimiter=';')

    # Spaltennamen bereinigen
    data.columns = [col.strip() for col in data.columns]

    # Nur relevante Spalten auswählen
    data = data[['MESS_DATUM', 'RWS_10']]

    # RWS_10 in numerischen Wert umwandeln und fehlerhafte Werte (-999) als 0 behandeln
    data['RWS_10'] = pd.to_numeric(data['RWS_10'], errors='coerce').replace(-999, 0)

    # Datum in einen datetime-Typ umwandeln
    data['time'] = pd.to_datetime(data['MESS_DATUM'], format='%Y%m%d%H%M')

    # Index setzen
    data.set_index('time', inplace=True)

    # 1-stündige und 6-stündige rollende Summen berechnen
    data['rolling_1h_sum'] = data['RWS_10'].rolling('1h').sum()
    data['rolling_6h_sum'] = data['RWS_10'].rolling('6h').sum()

    # Klassifikation basierend auf den Regeln
    def classify(row):
        if row['rolling_6h_sum'] > 35 or row['rolling_1h_sum'] > 25:
            return "High"  # Unwetter
        elif 20 <= row['rolling_6h_sum'] <= 35 or 15 <= row['rolling_1h_sum'] <= 25:
            return "Medium"  # Markantes Wetter
        else:
            return "Low"  # Keine besonderen Bedingungen

    data['classification'] = data.apply(classify, axis=1)

    # Tagesklassifikation berechnen
    daily_classification = (
        data.groupby(data.index.date)['classification']
        .apply(lambda x: "High" if "High" in x.values else ("Medium" if "Medium" in x.values else "Low"))
    )

    # DataFrame mit Tagesklassifikation erstellen
    daily_classification_df = daily_classification.reset_index()
    daily_classification_df.columns = ['date', 'daily_classification']

    # Ergebnisse speichern
    return data[['RWS_10', 'rolling_1h_sum', 'rolling_6h_sum', 'classification']], daily_classification_df


# Datei analysieren
file_path = '/Users/paulgraefe/PycharmProjects/scientificProject/data-processing/produkt_zehn_min_rr_20230524_20241123_05992.csv'
classified_data, daily_data = classify_rainfall_daily(file_path)

# Ergebnisse speichern
classified_data.to_csv('/Users/paulgraefe/PycharmProjects/scientificProject/data-processing/classified_rainfall.csv')
daily_data.to_csv('/Users/paulgraefe/PycharmProjects/scientificProject/data-processing/daily_rainfall_classification.csv')

print("Die detaillierten Klassifikationen und die Tagesklassifikationen wurden gespeichert.")
# Count occurrences of each state
state_counts = daily_data['daily_classification'].value_counts()

# Print the counts
print(state_counts)

