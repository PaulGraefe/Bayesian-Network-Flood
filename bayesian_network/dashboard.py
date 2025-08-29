import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pgmpy.inference import VariableElimination

from utils import get_exact_inference_one_state, perform_sensitivity_analysis
from bn import build_bn
from variables import *
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns



# ----------------------------------
# Titel & Einführung
# ----------------------------------
# ---- Seite und Titel ----
st.set_page_config(
    page_title="🌊 Flutrisiko-Dashboard",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown("""
Dieses Dashboard visualisiert die Bewertung von Flutrisiken landwirtschaftlicher Flächen auf Basis
eines Bayesschen Netzes (BN) kombiniert mit GIS-Analysen.

**Fokus:**
- Bewertung des Flutrisikos nach Starkregenereignissen
- Sensitivitätsanalyse und Einflussvariablen
- Validierung anhand realer Flurstücksdaten
""")

# ----------------------------------
# Variable Overview
# ----------------------------------
st.header("Variablenübersicht")

variables = [
    RAINFALL_AMOUNT,
    RAINFALL_INTENSITY,
    TEMPERATURE,
    LAND_USE,
    SOIL_MOISTURE,
    SOIL_TYPE,
    RUNOFF_COEFFICIENT,
    ELEVATION,
    SLOPE,
    HAZARD,
    RIVER_DISCHARGE,
    RIVER_EXPOSURE,
    STREET_DENSITY,
    PROXIMITY_TO_FOREST,
    VULNERABILITY,
    PROXIMITY_TO_RIVER,
    EXPOSURE,
    FLOOD_RISK
]


state_names_dictionary = {
    RAINFALL_INTENSITY: ['High', 'Medium', 'Low'],
    RAINFALL_AMOUNT: ['High', 'Medium', 'Low'],
    TEMPERATURE: ['High', 'Medium', 'Low'],
    LAND_USE: ['Greenland', 'Farmland'],
    SOIL_MOISTURE: ['High', 'Medium', 'Low'],
    SOIL_TYPE: ['L', 'LT', 'sL', 'T', 'Unknown'],
    RUNOFF_COEFFICIENT: ['High', 'Medium', 'Low'],
    ELEVATION: ['High', 'Medium', 'Low'],
    SLOPE: ['High', 'Medium', 'Low'],
    HAZARD: ['High', 'Medium', 'Low'],
    RIVER_DISCHARGE: ['High', 'Medium', 'Low'],
    RIVER_EXPOSURE: ['High', 'Medium', 'Low'],
    PROXIMITY_TO_RIVER: ['High', 'Medium', 'Low'],
    STREET_DENSITY: ['High', 'Medium', 'Low'],
    PROXIMITY_TO_FOREST: ['High', 'Medium', 'Low'],
    VULNERABILITY: ['High', 'Medium', 'Low'],
    EXPOSURE: ['High', 'Medium', 'Low'],
    FLOOD_RISK: ['Yes', 'No']
}

# Tabelle anzeigen
state_table = pd.DataFrame.from_dict(state_names_dictionary, orient='index').transpose()
st.dataframe(state_table)

st.markdown("""
Die Tabelle zeigt alle Variablen des Bayesschen Netzes und deren mögliche Zustände.
Diese bilden die Grundlage für die Szenarioanalyse und die Risikobewertung.
""")

# BN laden
model = build_bn()
infer = VariableElimination(model)

st.title("🌊 Flood Risk Szenarioanalyse")

st.markdown("Wähle die Szenariowerte für die Eingangsvariablen:")

# Dropdowns für feste Werte
rainfall = st.selectbox("Regenintensität", ["Low", "Medium", "High"], index=2)
temperature = st.selectbox("Temperatur", ["Low", "Medium", "High"], index=1)
soil = st.selectbox("Bodenfeuchte", ["Low", "Medium", "High"], index=2)
river = st.selectbox("Abfluss", ["Low", "Medium", "High"], index=2)

# Evidenz aus den Dropdowns zusammenstellen
fixed_values = {
    "RAINFALL_INTENSITY": rainfall,
    "TEMPERATURE": temperature,
    "SOIL_MOISTURE": soil,
    "RIVER_DISCHARGE": river
}
st.title("🌊 Flutrisiko-Dashboard")
st.markdown("Berechnung des Flutrisikos für viele Flurstücke auf Basis einer CSV-Datei.")

# 3. Pfade im Projektbaum (statt Upload)
input_file = "bayesian_network/InferenceData/flst_final.csv"
output_file = "bayesian_network/InferenceData/output_with_risk.csv"


# CSV laden
df = pd.read_csv(input_file, delimiter=';')
st.write("✅ Datei erfolgreich geladen. Vorschau:")
st.dataframe(df.head())

# Berechnung starten
if st.button("Flutrisiko berechnen"):
    results = []
    for index, row in df.iterrows():
        evidence = {
            'ELEVATION': row['ELEVATION'],
            'SLOPE': row['SLOPE'],
            'PROXIMITY_TO_RIVER': row['PROXIMITY_TO_RIVER'],
            'PROXIMITY_TO_FOREST': row['PROXIMITY_TO_FOREST'],
            'STREET_DENSITY': row['STREET_DENSITY'],
            'LAND_USE': row['LAND_USE'],
            'SOIL_TYPE': row['SOIL_TYPE']
        }

        combined_evidence = {**evidence, **fixed_values}

        probability_yes = get_exact_inference_one_state(
            "FLOOD_RISK", infer, combined_evidence
        )

        results.append({
            'oid': row['oid'],
            'FLOOD_RISK_Yes_Probability': probability_yes
        })

    # Ergebnisse speichern
    result_df = pd.DataFrame(results)


    # Stil setzen
    sns.set(style="whitegrid")  # weißer Hintergrund, feines Grid
    plt.rcParams.update({
        "font.size": 14,
        "figure.figsize": (10, 6),
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    fig, ax = plt.subplots()

    # Nur Histogramm ohne KDE
    sns.histplot(
        result_df['FLOOD_RISK_Yes_Probability'],
        bins=20,
        color="royalblue",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
        ax=ax
    )

    # ---- Histogramm Flutrisiko ----
    st.subheader("📊 Verteilung der Flutrisiken")
    fig, ax = plt.subplots(figsize=(4, 4))  # hochkant, kompakt
    sns.histplot(
        result_df['FLOOD_RISK_Yes_Probability'],
        bins=20,
        color="royalblue",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
        ax=ax
    )
    ax.set_xlabel("Flutrisiko-Wahrscheinlichkeit (Yes)", fontsize=10)
    ax.set_ylabel("Anzahl Flurstücke", fontsize=10)
    ax.set_title("Verteilung des Flutrisikos", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.25)

    st.pyplot(fig, bbox_inches='tight')  # nutzt den verfügbaren Platz effizient

    st.subheader("🔍 Sensitivitätsanalyse")

    # Variablen, deren Einfluss auf FLOOD_RISK untersucht werden soll
    components_to_analyze = [
        "HAZARD", "VULNERABILITY", "EXPOSURE", "RIVER_EXPOSURE",
        "RUNOFF_COEFFICIENT", "RAINFALL_AMOUNT"
    ]

    target_variable = 'FLOOD_RISK'

    # Sensitivitätsanalyse ausführen
    # (Leere Evidenz {} bedeutet, dass alle Variablen unvoreingestellt sind)
    sensitivity_results = perform_sensitivity_analysis(
        target_variable, infer, {}, components_to_analyze, model
    )

    # Aggregiere nach Variable: Differenz zwischen Max und Min Wahrscheinlichkeit von 'Yes'
    sensitivity_summary = sensitivity_results.groupby("Variable").apply(
        lambda df: df["Target_Probabilities"].apply(lambda x: x[0]).max() -
                   df["Target_Probabilities"].apply(lambda x: x[0]).min()
    ).reset_index()
    sensitivity_summary.columns = ["Variable", "Sensitivity"]

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 4))  # kompakt und hochkant
    sns.barplot(data=sensitivity_summary, x="Sensitivity", y="Variable", palette="viridis", ax=ax)
    ax.set_xlabel("Delta FLOOD_RISK (Yes)")
    ax.set_ylabel("Variable")
    ax.set_title("Sensitivität der Variablen auf FLOOD_RISK")
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)