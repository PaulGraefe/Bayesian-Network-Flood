import pandas as pd
from datetime import date, timedelta

# 1. Lade die bestehende CSV-Datei mit den dynamischen Wetterdaten
df_weather = pd.read_csv('/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/classified_daily_summary.csv')

# Stelle sicher, dass das Datum in den Wetterdaten als Datumsobjekt vorliegt
df_weather['DATE'] = pd.to_datetime(df_weather['DATE'])  # Ersetze 'DATE' mit dem tatsächlichen Spaltennamen

# 2. Definiere die Szenarien mit festen Werten
scenarios = [
    {'LAND_USE': 'Grassland', 'SOIL_TYPE': 'T', 'ELEVATION': 'High', 'SLOPE': 'Steep', 'RUNOFF_COEFFICIENT': 'High',
     'RIVER_DENSITY': 'Low', 'FOREST_DENSITY': 'High', 'STREET_DENSITY': 'Low', 'FLOOD_RISK': 'Low',
     'RIVER_EXPOSURE': 'Low', 'EXPOSURE': 'Low', 'HAZARD': 'Low', 'VULNERABILITY': 'Low'},

    {'LAND_USE': 'Grassland', 'SOIL_TYPE': 'L', 'ELEVATION': 'High', 'SLOPE': 'Steep', 'RUNOFF_COEFFICIENT': 'Medium',
     'RIVER_DENSITY': 'Low', 'FOREST_DENSITY': 'High', 'STREET_DENSITY': 'Medium', 'FLOOD_RISK': 'Low',
    'RIVER_EXPOSURE': 'Low', 'EXPOSURE': 'Low', 'HAZARD': 'Low', 'VULNERABILITY': 'Low'},

    {'LAND_USE': 'Farmland', 'SOIL_TYPE': 'LT', 'ELEVATION': 'Medium', 'SLOPE': 'Moderate', 'RUNOFF_COEFFICIENT': 'High',
     'RIVER_DENSITY': 'Low', 'FOREST_DENSITY': 'Low', 'STREET_DENSITY': 'Low', 'FLOOD_RISK': 'Medium',
    'RIVER_EXPOSURE': 'Low', 'EXPOSURE': 'Low', 'HAZARD': 'Low', 'VULNERABILITY': 'Low'},

    {'LAND_USE': 'Grassland', 'SOIL_TYPE': 'T', 'ELEVATION': 'Medium', 'SLOPE': 'Moderate', 'RUNOFF_COEFFICIENT': 'Medium',
     'RIVER_DENSITY': 'Low', 'FOREST_DENSITY': 'Low', 'STREET_DENSITY': 'Medium', 'FLOOD_RISK': 'Medium',
    'RIVER_EXPOSURE': 'Low', 'EXPOSURE': 'Low', 'HAZARD': 'Low', 'VULNERABILITY': 'Low'},

    {'LAND_USE': 'Farmland', 'SOIL_TYPE': 'L', 'ELEVATION': 'Low', 'SLOPE': 'Flat', 'RUNOFF_COEFFICIENT': 'Medium',
     'RIVER_DENSITY': 'High', 'FOREST_DENSITY': 'Low', 'STREET_DENSITY': 'High', 'FLOOD_RISK': 'High',
    'RIVER_EXPOSURE': 'Low', 'EXPOSURE': 'Low', 'HAZARD': 'Low', 'VULNERABILITY': 'Low'},

    {'LAND_USE': 'Farmland', 'SOIL_TYPE': 'sL', 'ELEVATION': 'Low', 'SLOPE': 'Flat', 'RUNOFF_COEFFICIENT': 'Low',
     'RIVER_DENSITY': 'High', 'FOREST_DENSITY': 'Low', 'STREET_DENSITY': 'High', 'FLOOD_RISK': 'High',
    'RIVER_EXPOSURE': 'Low', 'EXPOSURE': 'Low', 'HAZARD': 'Low', 'VULNERABILITY': 'Low'},
]

# 3. Erstelle ein leeres DataFrame für die konsolidierten Daten
consolidated_df = pd.DataFrame()

# 4. Definiere Startdatum und filtere Wetterdaten ab diesem Datum
start_date = date(2023, 7, 31)
filtered_weather = df_weather[df_weather['DATE'] >= pd.Timestamp(start_date)]

# Simuliere für jedes Szenario die Wetterdaten für eine bestimmte Anzahl von Tagen
days_per_scenario = 367

for scenario in scenarios:
    # Nimm nur die ersten `days_per_scenario` Tage ab dem Startdatum
    scenario_weather = filtered_weather.head(days_per_scenario).copy()

    # Füge die festen Werte aus dem Szenario hinzu
    for key, value in scenario.items():
        scenario_weather[key] = value

    # Setze Flood_Risk überall auf 'Low', falls initial
    scenario_weather['FLOOD_RISK'] = 'Low'

    # Hänge die Daten des Szenarios an das konsolidierte DataFrame an
    consolidated_df = pd.concat([consolidated_df, scenario_weather], ignore_index=True)

# 5. Speichere die konsolidierte Datei
consolidated_df.to_csv('/Users/paulgraefe/PycharmProjects/scientificProject/data/open-meteo-data/consolidated_with_scenarios.csv',
                       index=False)

print("Konsolidierte Daten erfolgreich gespeichert.")
