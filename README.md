# 🌊 Bayesian Network Flood Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?logo=streamlit)](https://bayesian-network-flood-wkfuzyg7i3dq3yhhxtaeom.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

Ein Projekt zur **Vorhersage von Flutrisiken** auf Basis von **Bayesschen Netzwerken**.  
Das Modell kombiniert **Geodaten**, **zeitlich veränderliche Daten** und **statische Variablen**, um für jedes Flurstück eine **Flutrisikowahrscheinlichkeit** zu berechnen.

🌐 **Live Dashboard:**  
[Hier klicken, um das Streamlit-Dashboard zu starten](https://bayesian-network-flood-wkfuzyg7i3dq3yhhxtaeom.streamlit.app/)

---

## 🚀 Features

- **Bayessches Netzwerk** zur probabilistischen Modellierung von Flutrisiken  
- Verarbeitung und Klassifizierung von **Eingangsdaten** (CSV-Dateien)  
- Dynamische **Wahrscheinlichkeitsberechnungen pro Flurstück**  
- Sensitivitätsanalysen zur Bewertung der Einflussfaktoren  
- Interaktive **Streamlit-Web-App** zur Visualisierung der Ergebnisse  

---

## 🗂 Projektstruktur


---

## 📜 Beschreibung der Kernskripte

| Datei                | Beschreibung |
|----------------------|--------------|
| **bn.py**            | Implementiert das Bayessche Netzwerk. Berechnet Flutrisikowahrscheinlichkeiten und speichert diese in einer neuen CSV-Datei. |
| **dashboard.py**     | Startet das Streamlit Dashboard zur Visualisierung der Berechnungen und Ergebnisse. |
| **extended_classes.py** | Erweiterungen für Bayessche Netzwerke, inkl. Weighted Likelihood und Approximationstechniken (z.B. Rejection Sampling). |
| **utils.py**         | Hilfsfunktionen für Risikoanalysen, Sensitivitätsanalysen und Visualisierungen. |
| **variables.py**     | Definition der relevanten Variablen, die im Bayesschen Netzwerk genutzt werden. |
| **InferenceData/**   | Enthält Eingabe- und Ausgabedateien für die Risiko-Berechnung pro Flurstück. |

---

## 📊 Live-Dashboard

Das Dashboard ermöglicht die **interaktive Exploration** der Ergebnisse:

- Anzeige des **berechneten Flutrisikos** pro Flurstück  
- Vergleich verschiedener Variablen und Einflussfaktoren  
- Möglichkeit, eigene Eingabedaten hochzuladen (optional)

🔗 **Direkter Link zum Dashboard:**  
[https://bayesian-network-flood-wkfuzyg7i3dq3yhhxtaeom.streamlit.app/](https://bayesian-network-flood-wkfuzyg7i3dq3yhhxtaeom.streamlit.app/)

---

## ⚙️ Installation & Setup

### 1️⃣ Repository klonen
```bash
git clone https://github.com/DEIN_USERNAME/Bayesian-Network-Flood.git
cd Bayesian-Network-Flood
