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
    HAZARD: ['High', 'Low'],  # -
    RIVER_DISCHARGE: ['High', 'Medium', 'Low'],
    RIVER_EXPOSURE: ['High', 'Low'],
    PROXIMITY_TO_RIVER: ['High', 'Medium', 'Low'],
    STREET_DENSITY: ['High', 'Medium', 'Low'],
    PROXIMITY_TO_FOREST: ['High', 'Medium', 'Low'],
    VULNERABILITY: ['High', 'Low'],  # -
    EXPOSURE: ['High', 'Low'],
    FLOOD_RISK: ['High', 'Low']
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
        [2561 / 3659],
        [1098 / 3659]
    ],

    SOIL_MOISTURE: [
        [121 / 365],
        [124 / 365],
        [120 / 365]
    ],

    SOIL_TYPE: [
        [404 / 3659],
        [271 / 3659],
        [136 / 3659],
        [1747 / 3659],
        [1101 / 3659]
    ],

    ELEVATION: [
        [1129 / 3659],
        [1757 / 3659],
        [773 / 3659]
    ],

    SLOPE: [
        [2207 / 3659],
        [689 / 3659],
        [763 / 3659]
    ],

    STREET_DENSITY: [
        [1454 / 3659],
        [1647 / 3659],
        [558 / 3659]
    ],

    PROXIMITY_TO_FOREST: [
        [2413 / 3659],
        [653 / 3659],
        [593 / 3659]
    ],

    RUNOFF_COEFFICIENT: [
        [0.85, 0.75, 0.5, 0.99, 0.75,   0.6, 0.65, 0.35, 0.9, 0.65,  0.4, 0.45, 0.2, 0.85, 0.45,  0.9, 0.95, 0.8, 0.99, 0.95,       0.65, 0.7, 0.55, 0.85, 0.7,     0.45, 0.5, 0.35, 0.6, 0.5],
        [0.1, 0.15, 0.3, 0.01, 0.15,   0.3, 0.25, 0.35, 0.1, 0.25,  0.3, 0.35, 0.3, 0.1, 0.35,  0.075, 0.05, 0.15, 0.01, 0.05,     0.25, 0.25, 0.35, 0.125, 0.25,   0.25, 0.3, 0.3, 0.3, 0.3],
        [0.05, 0.1, 0.2, 0.0, 0.1,     0.1, 0.1, 0.3, 0.0, 0.1,      0.3, 0.2, 0.5, 0.05, 0.2,   0.025, 0.0, 0.05, 0.0, 0.0,        0.1,  0.05, 0.1, 0.025, 0.05,    0.3, 0.2, 0.35, 0.1, 0.2]
    ],

    RIVER_DISCHARGE: [
        [72 / 365],
        [187 / 365],
        [106 / 365]
    ],

    PROXIMITY_TO_RIVER: [
        [3207 / 3659],
        [226 / 3659],
        [226 / 3659]
    ],

    HAZARD: [
        [0.6, 0.99, 0.7,  0.4, 0.55, 0.45,  0.15, 0.2, 0.1],
        [0.4, 0.01, 0.3,  0.6, 0.45, 0.55,  0.85, 0.8, 0.9]
    ],

    VULNERABILITY: [
        [0.01, 0.15, 0.25,  0.15, 0.35, 0.45,  0.4, 0.7, 0.99,  0.1, 0.2, 0.3,   0.15, 0.3, 0.4,   0.25, 0.4, 0.75,   0.05, 0.1, 0.3,   0.075, 0.2, 0.3,       0.15, 0.35, 0.55],
        [0.99, 0.85, 0.75,  0.85, 0.65, 0.55,  0.6, 0.3, 0.01,   0.9, 0.8, 0.7,   0.85, 0.7, 0.6,   0.75, 0.6, 0.25,   0.95, 0.9, 0.7,   0.925, 0.8, 0.7,       0.85, 0.65, 0.45]

    ],

    RIVER_EXPOSURE: [
        [0.1, 0.45, 0.99, 0.05, 0.3, 0.4, 0.01, 0.1, 0.2],
        [0.9, 0.55, 0.01, 0.95, 0.7, 0.6, 0.99, 0.9, 0.8]
    ],

    EXPOSURE: [
        [0.6, 0.7, 0.99,   0.4, 0.45, 0.7,  0.35, 0.4, 0.45,  0.25, 0.2, 0.55,    0.15, 0.225, 0.275,    0.01, 0.1, 0.15],
        [0.4, 0.3, 0.01,   0.6, 0.55, 0.3,  0.65, 0.6, 0.55,   0.75, 0.8, 0.45,   0.85, 0.775, 0.725,    0.99, 0.9, 0.85]

    ],

    FLOOD_RISK: [
        [0.99, 0.75, 0.75, 0.4,   0.15, 0.05, 0.05, 0.01],
        [0.01, 0.25, 0.25, 0.6,   0.85, 0.95, 0.95, 0.99]
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

evidence = {
    'RAINFALL_INTENSITY': 'High',
    'TEMPERATURE': 'Medium',
    'SOIL_MOISTURE': 'High',
    'RIVER_DISCHARGE': 'High',
    'ELEVATION': 'High',
    'SLOPE': 'High',
    'PROXIMITY_TO_RIVER': 'High',
    'PROXIMITY_TO_FOREST': 'High',
    "STREET_DENSITY": 'Low',
    'LAND_USE': "Greenland",
    "SOIL_TYPE": "Unknown"
}

lowest_flst = {
    'RAINFALL_INTENSITY': 'High',
    'TEMPERATURE': 'Medium',
    'SOIL_MOISTURE': 'High',
    'RIVER_DISCHARGE': 'High',
    'ELEVATION': 'High',
    'SLOPE': 'High',
    'PROXIMITY_TO_RIVER': 'High',
    'PROXIMITY_TO_FOREST': 'High',
    "STREET_DENSITY": 'Low',
    'LAND_USE': "Greenland",
    "SOIL_TYPE": "Unknown"
}

highest_flst = {
    'RAINFALL_INTENSITY': 'High',
    'TEMPERATURE': 'Medium',
    'SOIL_MOISTURE': 'High',
    'RIVER_DISCHARGE': 'High',
    'ELEVATION': 'Low',
    'SLOPE': 'Low',
    'PROXIMITY_TO_RIVER': 'Low',
    'PROXIMITY_TO_FOREST': 'Low',
    "STREET_DENSITY": 'Medium',
    'LAND_USE': "Greenland",
    "SOIL_TYPE": "T"
}

evidence3 = {}

print("simple example")
print(get_exact_inference_one_state("FLOOD_RISK", exact_infer, lowest_flst))
print(get_exact_inference_one_state("FLOOD_RISK", exact_infer, highest_flst))

variables_to_analyze = ['RAINFALL_INTENSITY', 'TEMPERATURE', 'SOIL_MOISTURE', 'RIVER_DISCHARGE', 'ELEVATION', 'SLOPE',
                        'PROXIMITY_TO_RIVER', 'PROXIMITY_TO_FOREST', 'STREET_DENSITY', 'LAND_USE', 'SOIL_TYPE']

components_to_analyze = ["HAZARD", "VULNERABILITY", "EXPOSURE", "RIVER_EXPOSURE", "RUNOFF_COEFFICIENT", "RAINFALL_AMOUNT"]
analyze_exposure = ["RIVER_EXPOSURE", "PROXIMITY_TO_RIVER", "RIVER_DISCHARGE", "STREET_DENSITY", "PROXIMITY_TO_FOREST"]

analyze_vulnerability = ['SOIL_MOISTURE', 'ELEVATION', 'SLOPE', 'LAND_USE', 'SOIL_TYPE', "RUNOFF_COEFFICIENT"]

analyze_hazard = ["RAINFALL_INTENSITY", "RAINFALL_AMOUNT", "TEMPERATURE", "HAZARD"]
target_variable = 'EXPOSURE'

sensitivity_results = perform_sensitivity_analysis(target_variable, exact_infer, evidence3, analyze_exposure, model)
#print(sensitivity_results)
plot_sensitivity_results(sensitivity_results)

variable_pair = ("ELEVATION", "SLOPE")

# Wechselwirkungen analysieren
interaction_results = analyze_variable_interaction(
    target_variable="FLOOD_RISK",
    inference=exact_infer,
    variable_pair=variable_pair,
    model=model,
    evidence=evidence
)
print(interaction_results)

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
