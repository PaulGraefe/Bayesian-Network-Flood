import pandas as pd

# Liste der klassifizierten Dateien und relevante Spalten
classified_files = [
    ("/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/rain_sum_classified.csv", "rainfall_amount_class"),
    ("/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/rain_intensity_classified.csv", "rain_intensity_class"),
    ("/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/river_discharge_classified.csv", "river_discharge_class"),
    ("/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/soil_moisture_classified.csv", "soil_moisture_class"),
    ("/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/temperature_classified.csv", "temperature_class")
]

# Initialisiere einen leeren DataFrame für den Merge
combined_df = None

# Verarbeite jede Datei
for file_path, class_column in classified_files:
    try:
        # Lade die Datei
        df = pd.read_csv(file_path)

        # Reduziere auf `date` und die Klassifikationsspalte
        if 'date' not in df.columns:
            raise ValueError(f"Column 'date' not found in {file_path}")

        df = df[['date', class_column]]

        # Merge mit dem kombinierten DataFrame
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='date', how='outer')

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Speichere die kombinierte Datei
if combined_df is not None:
    output_csv_path = "/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/classified_daily_summary.csv"
    combined_df.to_csv(output_csv_path, index=False)
    print(f"Combined classified data saved to: {output_csv_path}")
else:
    print("No valid dataframes to combine.")