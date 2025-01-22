# Bayesian-Network-Flood


## bayesian_network

### bn.py:

Diese Python-Datei implementiert ein Bayessches Netzwerk,
um das Flutrisiko basierend auf datenbasierten Variablen. Die Datei nutzt eine Kombination aus 
festgelegten Eingabewerten und Geodaten, um Wahrscheinlichkeiten
für Flutrisiken zu berechnen, und wendet diese Berechnungen auf
jede Zeile einer CSV-Datei an, die Flurstückdaten enthält.
Abschließend werden die Ergebnisse inklusive der berechneten 
Flutrisikoprobabilität in einer neuen CSV-Datei gespeichert.



### extended_classes.py:
Dieses Skript erweitert bestehende Bayessche Netzwerke durch die Möglichkeit, Daten mithilfe gewichteter Likelihood zu simulieren. Zusätzlich ermöglicht es die Verwendung von Approximationstechniken wie gewichtetes oder Ablehnungssampling für probabilistische Abfragen. 



### utlis.py:

Dieses Skript enthält Funktionen für die Anwendung und Analyse Bayesscher Netzwerke. Es ermöglicht die Berechnung von Flutrisiken basierend auf spezifischen Eingabedaten und unterstützt Sensitivitätsanalysen, um den Einfluss einzelner Variablen zu bewerten. Zudem bietet es Werkzeuge zur Erstellung und Visualisierung von bedingten Wahrscheinlichkeitsverteilungen sowie zur Diskretisierung von Daten in definierte Zustände.


### variables.py

implementiert die Variablen, die für die Analyse mit dem BN relevant sind


### plotHistogramm.py

erzeugt ein Histogramm basierend auf den Eingabe CSV-Datensatz 

### residuals.py

Pythpon.Skript welches die Residuen und den RMSE auf Basis der Validierung berechnet um ein Genauigkeitsmaß für das BN zu erhalten
### InferenceData

enthält die Eingaben- und Ausgaben CSV-Datensätze fpr die Flutriskoberechnung der Flurstücke
## data

enhält allle relevanten  CSV-Datensätze für sich zeitlich verändernde Variablen

## data-processing

enhält alle Datenprozessierungsskripte mit denen die  CSV-Datensätze aufbereitet und klassifiziert wurden
