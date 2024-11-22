import pandas as pd

# Funktion zur Verarbeitung und Aggregation von Daten
def process_weather_data(soil_moisture_file, rain_file, soil_output_path, rain_output_path):
    # Lade die CSV-Dateien
    data_1 = pd.read_csv(soil_moisture_file)
    data_2 = pd.read_csv(rain_file)

    # Daten bereinigen und vorbereiten
    data_1.columns = data_1.iloc[1]
    data_1 = data_1[2:]
    data_1 = data_1.rename(columns={"time": "datetime", "soil_moisture_0_to_7cm (m³/m³)": "soil_moisture"})
    data_1 = data_1[["datetime", "soil_moisture"]].dropna()

    data_2.columns = data_2.iloc[1]
    data_2 = data_2[2:]
    data_2 = data_2.rename(columns={"time": "datetime", "rain (mm)": "rain"})
    data_2 = data_2[["datetime", "rain"]].dropna()

    # Konvertiere die Datumswerte in Pandas-Datetime-Objekte
    data_1["datetime"] = pd.to_datetime(data_1["datetime"])
    data_2["datetime"] = pd.to_datetime(data_2["datetime"])

    # Füge eine "date"-Spalte hinzu, um nach Tagen zu gruppieren
    data_1["date"] = data_1["datetime"].dt.date
    data_2["date"] = data_2["datetime"].dt.date

    # Konvertiere die Werte in numerische Typen
    data_1["soil_moisture"] = pd.to_numeric(data_1["soil_moisture"], errors='coerce')
    data_2["rain"] = pd.to_numeric(data_2["rain"], errors='coerce')

    # Gruppiere nach Datum und berechne die gewünschten Werte
    daily_soil_moisture = data_1.groupby("date")["soil_moisture"].mean().reset_index()
    daily_rain = data_2.groupby("date")["rain"].max().reset_index()

    # Speichere die aggregierten Daten als neue CSV-Dateien
    daily_soil_moisture.to_csv(soil_output_path, index=False)
    daily_rain.to_csv(rain_output_path, index=False)

    print(f"Dateien gespeichert:\n- Bodenfeuchtigkeit: {soil_output_path}\n- Regenintensität: {rain_output_path}")

# Beispielaufruf
soil_moisture_file = '/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/hourly/soil_moisture_hourly.csv'  # Pfad zur Datei mit Bodenfeuchtigkeitsdaten
rain_file = '/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/hourly/rain_hourly.csv'  # Pfad zur Datei mit Regenintensitätsdaten
soil_output_path = '/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/soil_moisture.csv'  # Speicherpfad für Bodenfeuchtigkeit
rain_output_path = '/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/rain_intensity.csv'  # Speicherpfad für Regenintensität

process_weather_data(soil_moisture_file, rain_file, soil_output_path, rain_output_path)
