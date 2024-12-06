from readline import redisplay
import warnings

from pgmpy.estimators import MaximumLikelihoodEstimator

# Suppress pgmpy internal deprecated use of third party libraries.
warnings.simplefilter(action='ignore', category=FutureWarning)
# Suppress UserWarning related to machine precision calculations of percentage.
warnings.simplefilter(action='ignore', category=UserWarning)

# Import internal modules.
from utils import *
from variables import *
from extended_classes import *
from pgmpy.utils import get_example_model


# Import pgmpy modules.
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination, ApproxInference
from pgmpy.sampling import GibbsSampling

# Import graphics related libraries and modules.
import matplotlib.pyplot as plt
# from IPython.display import display

# Import other useful librares and modules.
import numpy as np
import pandas as pd
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
print(f"Variables: {'; '.join(variables)}")

state_names_dictionary = {
    RAINFALL_INTENSITY: ['High', 'Medium', 'Low'],  # -
    RAINFALL_AMOUNT: ['High', 'Medium', 'Low'],  # -
    TEMPERATURE: ['High', 'Medium', 'Low'],
    LAND_USE: ['Greenland', 'Farmland'],
    SOIL_MOISTURE: ['High', 'Medium', 'Low'],
    SOIL_TYPE: ['L', 'LT', 'sL', 'T', 'Unknown'],
    RUNOFF_COEFFICIENT: ['High', 'Medium', 'Low'],
    ELEVATION: ['High', 'Medium', 'Low'],  # -
    SLOPE: ['High', 'Medium', 'Low'],  # -
    HAZARD: ['High', 'Medium', 'Low'],  # -
    RIVER_DISCHARGE: ['High', 'Medium', 'Low'],
    RIVER_EXPOSURE: ['High', 'Medium', 'Low'],
    PROXIMITY_TO_RIVER: ['High', 'Medium', 'Low'],
    STREET_DENSITY: ['High', 'Medium', 'Low'],
    PROXIMITY_TO_FOREST: ['High', 'Medium', 'Low'],
    VULNERABILITY: ['High', 'Medium', 'Low'],  # -
    EXPOSURE: ['High', 'Medium', 'Low'],
    FLOOD_RISK: ['Yes', 'No']
}

evidence_dictionary = {
    RAINFALL_INTENSITY: None,
    RAINFALL_AMOUNT: [RAINFALL_INTENSITY],
    TEMPERATURE: None,
    HAZARD: [RAINFALL_AMOUNT, TEMPERATURE],
    LAND_USE: None,
    SOIL_MOISTURE: None,
    SOIL_TYPE: None,
    RUNOFF_COEFFICIENT: [LAND_USE, SOIL_MOISTURE, SOIL_TYPE],
    ELEVATION: None,
    SLOPE: None,
    VULNERABILITY: [RUNOFF_COEFFICIENT, ELEVATION, SLOPE],
    PROXIMITY_TO_RIVER: None,
    RIVER_DISCHARGE: None,
    RIVER_EXPOSURE: [RIVER_DISCHARGE, PROXIMITY_TO_RIVER],
    STREET_DENSITY: None,
    PROXIMITY_TO_FOREST: None,
    EXPOSURE: [RIVER_EXPOSURE, STREET_DENSITY, PROXIMITY_TO_FOREST],
    FLOOD_RISK: [HAZARD, VULNERABILITY, EXPOSURE]
}

edges = [
    (RAINFALL_INTENSITY, RAINFALL_AMOUNT),
    (RAINFALL_AMOUNT, HAZARD),
    (TEMPERATURE, HAZARD),
    (RAINFALL_AMOUNT, HAZARD),

    (LAND_USE, RUNOFF_COEFFICIENT),
    (SOIL_MOISTURE, RUNOFF_COEFFICIENT),
    (SOIL_TYPE, RUNOFF_COEFFICIENT),


    (RUNOFF_COEFFICIENT, VULNERABILITY),
    (ELEVATION, VULNERABILITY),
    (SLOPE, VULNERABILITY),

    (RIVER_DISCHARGE, RIVER_EXPOSURE),
    (PROXIMITY_TO_RIVER, RIVER_EXPOSURE),

    (RIVER_EXPOSURE, EXPOSURE),
    (STREET_DENSITY, EXPOSURE),
    (PROXIMITY_TO_FOREST, EXPOSURE),

    (HAZARD, FLOOD_RISK),
    (VULNERABILITY, FLOOD_RISK),
    (EXPOSURE, FLOOD_RISK)
]


values_dictionary = {
    RAINFALL_INTENSITY: [
        [1 / 365],
        [6 / 365],
        [358 / 365]
    ],

    RAINFALL_AMOUNT: [
        [0.9, 0.5, 0.01],
        [0.09, 0.3, 0.09],
        [0.01, 0.2, 0.9]
    ],

    TEMPERATURE: [
        [35 / 365],
        [174 / 365],
        [156 / 365]
    ],

    LAND_USE: [
        [0.65],
        [0.35]
    ],

    SOIL_MOISTURE: [
        [121 / 365],
        [124 / 365],
        [120 / 365]
    ],

    SOIL_TYPE: [
        [1 / 6],
        [1 / 6],
        [1 / 6],
        [1 / 4],
        [1 / 4]
    ],

    ELEVATION: [
        [1 / 3],
        [1 / 3],
        [1 / 3]
    ],

    SLOPE: [
        [1 / 3],
        [1 / 3],
        [1 / 3]
    ],

    STREET_DENSITY: [
        [1 / 6],
        [1 / 3],
        [1 / 2]
    ],

    PROXIMITY_TO_FOREST: [
        [3 / 6],
        [1 / 6],
        [2 / 6]
    ],

    RUNOFF_COEFFICIENT: [
        [0.75, 0.65, 0.4, 0.9, 1/3,   0.6, 0.65, 0.35, 0.7, 1/3,  0.5, 0.55, 0.3, 0.7, 1/3,  0.8, 0.85, 0.7, 0.9, 1/3,    0.65, 0.7, 0.55, 0.75,1/3,    0.45, 0.5, 0.35, 0.55, 1/3],
        [0.15, 0.25, 0.4, 0.09, 1/3, 0.3, 0.25, 0.35, 0.2, 1/3,  0.4, 0.35, 0.4, 0.2, 1/3,  0.15, 0.1, 0.2, 0.09, 1/3,   0.25, 0.25, 0.35, 0.125, 1/3,  0.25, 0.3, 0.3, 0.25, 1/3],
        [0.1, 0.1, 0.2, 0.01, 1/3,  0.1, 0.1, 0.3, 0.1, 1/3,    0.1, 0.1, 0.3, 0.1, 1/3,   0.05, 0.05, 0.1, 0.01, 1/3,  0.1, 0.05, 0.1, 0.125, 1/3,    0.3, 0.2, 0.35, 0.2, 1/3]
    ],

    RIVER_DISCHARGE: [
        [72 / 365],
        [187 / 365],
        [106 / 365]
    ],

    PROXIMITY_TO_RIVER: [
        [2 / 6],
        [1 / 6],
        [3 / 6]
    ],

    HAZARD: [
        [0.6, 0.9, 0.8, 0.3, 0.4, 0.3, 0.05, 0.1, 0.05],
        [0.3, 0.1, 0.1, 0.4, 0.4, 0.5, 0.15, 0.2, 0.15],
        [0.1, 0.0, 0.1, 0.3, 0.2, 0.2, 0.8, 0.7, 0.8]
    ],

    VULNERABILITY: [
        [0.1, 0.15, 0.3, 0.1, 0.3, 0.5, 0.2, 0.4, 0.75, 0.15, 0.25, 0.4, 0.2, 0.35, 0.5, 0.3, 0.4, 0.6, 0.01, 0.05, 0.2,
         0.05, 0.2, 0.3, 0.1, 0.25, 0.4],
        [0.2, 0.25, 0.3, 0.2, 0.5, 0.4, 0.3, 0.45, 0.2, 0.25, 0.35, 0.45, 0.35, 0.45, 0.4, 0.4, 0.5, 0.25, 0.09, 0.15,
         0.3, 0.15, 0.35, 0.4, 0.2, 0.35, 0.4],
        [0.7, 0.6, 0.4, 0.7, 0.2, 0.1, 0.5, 0.15, 0.05, 0.6, 0.4, 0.15, 0.45, 0.2, 0.1, 0.3, 0.1, 0.15, 0.9, 0.8, 0.5,
         0.8, 0.45, 0.3, 0.7, 0.4, 0.2]
    ],

    RIVER_EXPOSURE: [
        [0.1, 0.4, 0.9, 0.05, 0.2, 0.45, 0.01, 0.1, 0.2],
        [0.2, 0.4, 0.09, 0.35, 0.4, 0.35, 0.09, 0.3, 0.45],
        [0.7, 0.2, 0.01, 0.6, 0.4, 0.2, 0.9, 0.6, 0.35]
    ],

    EXPOSURE: [
        [0.5, 0.6, 0.9, 0.4, 0.45, 0.5, 0.35, 0.4, 0.45, 0.5, 0.45, 0.6, 0.3, 0.35, 0.45, 0.4, 0.45, 0.5, 0.35, 0.375,
         0.425, 0.275, 0.35, 0.4, 0.01, 0.2, 0.3],
        [0.4, 0.3, 0.09, 0.4, 0.4, 0.4, 0.3, 0.3, 0.4, 0.4, 0.4, 0.3, 0.5, 0.4, 0.4, 0.45, 0.4, 0.4, 0.35, 0.325, 0.3,
         0.35, 0.325, 0.3, 0.09, 0.2, 0.2],
        [0.1, 0.1, 0.01, 0.2, 0.15, 0.1, 0.35, 0.3, 0.15, 0.1, 0.15, 0.1, 0.2, 0.25, 0.15, 0.15, 0.15, 0.1, 0.3, 0.3,
         0.275, 0.375, 0.325, 0.3, 0.9, 0.6, 0.5]
    ],

    FLOOD_RISK: [
        [0.9, 0.8, 0.7, 0.85, 0.75, 0.65, 0.6, 0.55, 0.5, 0.8, 0.7, 0.6, 0.65, 0.55, 0.45, 0.5, 0.4, 0.3, 0.4, 0.3, 0.2,
         0.15, 0.1, 0.05, 0.1, 0.05, 0.01],
        [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.4, 0.45, 0.5, 0.2, 0.3, 0.4, 0.35, 0.45, 0.55, 0.5, 0.6, 0.7, 0.6, 0.7, 0.8,
         0.85, 0.9, 0.95, 0.9, 0.95, 0.99]
    ],
}




cpds = {v: get_tabular_cpd(v, state_names_dictionary, values_dictionary, evidence_dictionary) for v in variables}

for k, v in cpds.items():
    print('CPD Table for variable: {}'.format(k))
    # redisplay(cpd_to_pandas(v))
    print()

model = ExtendedBayesianNetwork(edges)

model.add_cpds(*[cpds[k] for k in cpds])

# plot_simple_bayesian_network(model)

exact_infer = VariableElimination(model)

# Feste Werte definieren
fixed_values = {
    'RAINFALL_INTENSITY': 'High',
    'TEMPERATURE': 'Medium',
    'SOIL_MOISTURE': 'High',
    'RIVER_DISCHARGE': 'High'
}

# CSV einlesen
input_file = '/Users/paulgraefe/PycharmProjects/scientificProject/bayesian_network/InterferenceData/flstkTEST.csv'
output_file = '/Users/paulgraefe/PycharmProjects/scientificProject/bayesian_network/InterferenceData/output_with_risk.csv'
df = pd.read_csv(input_file, delimiter=';')

# Ergebnis für jede Zeile berechnen
results = []
counter = 1
for index, row in df.iterrows():
    # Evidenz aus der CSV-Zeile extrahieren
    evidence = {
        'ELEVATION': row['ELEVATION'],
        'SLOPE': row['SLOPE'],
        'PROXIMITY_TO_RIVER': row['PROXIMITY_TO_RIVER'],
        'PROXIMITY_TO_FOREST': row['PROXIMITY_TO_FOREST'],
        'STREET_DENSITY': row['STREET_DENSITY'],
        'LAND_USE': row['LAND_USE'],
        'SOIL_TYPE': row['SOIL_TYPE']
    }

    # Feste Werte mit CSV-Werten kombinieren
    combined_evidence = {**evidence, **fixed_values}

    # Inferenz ausführen
    #result = exact_infer.query(variables=['FLOOD_RISK'], evidence=combined_evidence, show_progress=False)

    # Zielvariable
    target_variable = 'FLOOD_RISK'

    # Inferenz ausführen
    print(counter)
    counter = counter + 1
    #print_exact_inference(target_variable, exact_infer, evidence)
    print_exact_inference_one_state(target_variable, exact_infer, evidence)

    # Ergebnis speichern
    #flood_risk = result.values.argmax()  # Risiko basierend auf der höchsten Wahrscheinlichkeit
    #results.append({'oid': row['oid'], 'FLOOD_RISK': flood_risk})

# Ergebnisse als DataFrame speichern
#result_df = pd.DataFrame(results)

# Ergebnisse in eine neue CSV speichern
#result_df.to_csv(output_file, index=False, sep=';')

#print(f"Ergebnisse wurden in {output_file} gespeichert.")




variables_to_analyze = ['RAINFALL_INTENSITY', 'TEMPERATURE', 'SOIL_MOISTURE']

# Sensitivitätsanalyse ausführen
#sensitivity_results = perform_sensitivity_analysis(target_variable, exact_infer, evidence, variables_to_analyze)


'''
RAINFALL_FREQUENCY: ['Frequent', 'Medium', 'Rare'], #-
    RAINFALL_AMOUNT: ['Huge', 'Medium', 'Little'], #-
    ELEVATION: ['High', 'Medium', 'Low'],#-
    SLOPE: ['Steep', 'Flat'],#-
    HAZARD: ['High', 'Medium', 'Low'],#-
    AGRICULTURE_DENSITY: ['High', 'Medium', 'Low'],#-
    VULNERABILITY: ['High', 'Medium', 'Low'],#-
    RIVER_DENSITY: ['Dense', 'Sparse'],#-
    EXPOSURE: ['High', 'Medium', 'Low'],
    FLOOD_RISK: ['Yes', 'No']
'''
