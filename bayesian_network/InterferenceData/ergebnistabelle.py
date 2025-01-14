import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei einlesen
data = pd.read_csv("/Users/paulgraefe/PycharmProjects/scientificProject/bayesian_network/InterferenceData/mb2_output2.csv")

# Risiko in Kategorien einteilen
def categorize_risk(value):
    if value <= 0.5:
        return 'Low'
    else:
        return 'High'

# Risiko kategorisieren
data['Risk_Category'] = data['Flood'].apply(categorize_risk)

# Liste der relevanten Kategorien in der CSV
categories = ['SLOPE', 'PROXIMITY_TO_RIVER', 'PROXIMITY_TO_FOREST', 'STREET_DENSITY', 'ELEVATION', 'LAND_USE', 'SOIL_TYPE']

# Ergebnisse sammeln
results = {}

for category in categories:
    # Kreuztabelle für die aktuelle Kategorie und Risikokategorie erstellen
    pivot_table = pd.crosstab(data[category], data['Risk_Category'])
    results[category] = pivot_table

    # Kreuztabelle ausgeben
    print(f"Kreuztabelle für {category}:")
    print(pivot_table)
    print()

# Optional: Tabelle für alle Kategorien zusammenfügen
combined_table = pd.concat(results, axis=0)

# Ergebnisse anzeigen
print("Kombinierte Tabelle für alle Kategorien:")
print(combined_table)

# Optional: Speichern der kombinierten Tabelle als CSV
combined_table.to_csv("combined_risk_analysis.csv")

# Visualisierung jeder Kategorie
for category in categories:
    pivot_table = results[category]
    pivot_table.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Verteilung der Risikokategorien nach {category}')
    plt.xlabel(category)
    plt.ylabel('Anzahl Flurstücke')
    plt.legend(title='Risk Category')
    plt.tight_layout()
    plt.show()
