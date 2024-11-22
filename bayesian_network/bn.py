from readline import redisplay
import warnings

# Suppress pgmpy internal deprecated use of third party libraries.
warnings.simplefilter(action='ignore', category=FutureWarning)
# Suppress UserWarning related to machine precision calculations of percentage.
warnings.simplefilter(action='ignore', category=UserWarning)

# Import internal modules.
from utils import *
from variables import *
from extended_classes import *
# from graphics import *

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
    RAINFALL_FREQUENCY,
    ELEVATION,
    SLOPE,
    HAZARD,
    AGRICULTURE_DENSITY,
    VULNERABILITY,
    RIVER_DENSITY,
    EXPOSURE,
    FLOOD_RISK
]
print(f"Variables: {'; '.join(variables)}")

state_names_dictionary = {
    RAINFALL_FREQUENCY: ['Frequent', 'Medium', 'Rare'],  # -
    RAINFALL_AMOUNT: ['Huge', 'Medium', 'Little'],  # -
    ELEVATION: ['High', 'Medium', 'Low'],  # -
    SLOPE: ['Steep', 'Flat'],  # -
    HAZARD: ['High', 'Medium', 'Low'],  # -
    AGRICULTURE_DENSITY: ['High', 'Medium', 'Low'],  # -
    VULNERABILITY: ['High', 'Medium', 'Low'],  # -
    RIVER_DENSITY: ['Dense', 'Sparse'],  # -
    EXPOSURE: ['High', 'Medium', 'Low'],
    FLOOD_RISK: ['Yes', 'No']
}

evidence_dictionary = {
    RAINFALL_FREQUENCY: None,
    RAINFALL_AMOUNT: [RAINFALL_FREQUENCY],
    ELEVATION: None,
    SLOPE: [ELEVATION],
    HAZARD: [RAINFALL_AMOUNT, SLOPE],
    AGRICULTURE_DENSITY: None,
    VULNERABILITY: [AGRICULTURE_DENSITY],
    RIVER_DENSITY: None,
    EXPOSURE: [RIVER_DENSITY],
    FLOOD_RISK: [HAZARD, VULNERABILITY, EXPOSURE]
}

edges = [
    (RAINFALL_FREQUENCY, RAINFALL_AMOUNT),
    (ELEVATION, SLOPE),
    (RAINFALL_AMOUNT, HAZARD),
    (SLOPE, HAZARD),
    (AGRICULTURE_DENSITY, VULNERABILITY),
    (RIVER_DENSITY, EXPOSURE),
    (HAZARD, FLOOD_RISK),
    (VULNERABILITY, FLOOD_RISK),
    (EXPOSURE, FLOOD_RISK)
]

values_dictionary = {
    RAINFALL_FREQUENCY: [
        [0.15],  # Frequent
        [0.1],  # Medium
        [0.75]  # Rare
    ],
    RAINFALL_AMOUNT: [
        [0.7, 0.2, 0.1],
        [0.2, 0.5, 0.3],
        [0.1, 0.3, 0.6]
    ],
    ELEVATION: [
        [0.15],
        [0.1],
        [0.75]
    ],
    SLOPE: [
        [0.75, 0.6, 0.05],
        [0.25, 0.4, 0.95]
    ],
    HAZARD: [
        [0.2, 0.7, 0.1, 0.5, 0.1, 0.1],
        [0.2, 0.25, 0.3, 0.3, 0.3, 0.2],
        [0.6, 0.05, 0.6, 0.2, 0.6, 0.7]
    ],
    AGRICULTURE_DENSITY: [
        [0.3],
        [0.6],
        [0.1]
    ],
    VULNERABILITY: [
        [0.6, 0.3, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.2, 0.7]
    ],
    RIVER_DENSITY: [
        [0.6],
        [0.4]
    ],
    EXPOSURE: [
        [0.7, 0.05],
        [0.2, 0.25],
        [0.1, 0.7]
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
print('The model has been correctly developed: {}.'.format(model.check_model()))

# plot_simple_bayesian_network(model)

exact_infer = VariableElimination(model)

print_exact_inference(FLOOD_RISK, exact_infer)

# worst case
print_exact_inference(FLOOD_RISK, exact_infer,
                      evidence={RAINFALL_FREQUENCY: 'Frequent', RAINFALL_AMOUNT: 'Huge', ELEVATION: 'High',
                                SLOPE: 'Flat', HAZARD: 'High', AGRICULTURE_DENSITY: 'High', VULNERABILITY: 'High',
                                RIVER_DENSITY: 'Dense', EXPOSURE: 'High'
                                })

# random case
print_exact_inference(FLOOD_RISK, exact_infer, evidence={RAINFALL_FREQUENCY: 'Medium',
                                                         ELEVATION: 'High',
                                                         SLOPE: 'Steep',
                                                         AGRICULTURE_DENSITY: 'High',
                                                         RIVER_DENSITY: 'Dense',
                                                         })

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
