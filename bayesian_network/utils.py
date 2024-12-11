import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.factors.discrete.CPD import TabularCPD
from typing import Dict, List, Optional

from pgmpy.inference.ExactInference import VariableElimination

from extended_classes import ExtendedApproxInference


# Funktion zur Durchführung der Inferenz

def analyze_variable_interaction(target_variable, inference, variable_pair, model, evidence):
    var1, var2 = variable_pair
    states_var1 = model.get_cpds(var1).state_names[var1]
    states_var2 = model.get_cpds(var2).state_names[var2]

    results = []

    # Iteriere über alle Kombinationen der Zustände von var1 und var2
    for state1, state2 in itertools.product(states_var1, states_var2):
        evidence[var1] = state1
        evidence[var2] = state2

        # Führe Inferenz durch
        prob_dist = inference.query([target_variable], evidence=evidence)

        # Ergebnisse speichern
        results.append({
            "Variable1": var1,
            "State1": state1,
            "Variable2": var2,
            "State2": state2,
            "FLOOD_RISK_High": prob_dist.values[0]  # Annahme: High ist der erste Zustand
        })

    # Ursprüngliches Evidence wiederherstellen
    del evidence[var1]
    del evidence[var2]

    return pd.DataFrame(results)

''' 
Ermitteln der Zustände der Variablen:

Die Funktion holt die Zustände (z. B. „High“, „Medium“, „Low“) der analysierten Variablen über ihre CPDs.
Iterative Änderung der Evidenz:

Für jede Variable und jeden Zustand wird der Zustand in der Evidenz geändert. Die Evidenz beeinflusst die Zielvariable (Hazard, also FLOOD_RISK in Ihrem Fall).
Berechnung der Wahrscheinlichkeitsverteilung (Inference):

Die Funktion verwendet eine Inferenzmethode (z. B. Vorwärtsinferenz), um die Wahrscheinlichkeitsverteilung der Zielvariable (FLOOD_RISK) zu berechnen. Diese Verteilung basiert auf den aktuellen Werten der Evidenz und den CPDs des Modells.
Speichern der Ergebnisse:

Die Ergebnisse für jeden Zustand der analysierten Variablen werden gespeichert und später visualisiert.
'''
def perform_sensitivity_analysis(target_variable, inference, evidence, variables_to_analyze, model):
    results = []

    # Funktion, um die Zustände einer Variablen aus den CPDs zu extrahieren
    def get_variable_states(variable):
        cpd = model.get_cpds(variable)
        if cpd is None:
            raise ValueError(f"Keine CPD für die Variable {variable} im Modell gefunden.")
        return cpd.state_names[variable]

    # Berechnung der Wahrscheinlichkeiten ohne Evidenz
    no_evidence_result = inference.query([target_variable], evidence={})  # Keine Eingabe gegeben
    results.append({
        "Variable": "No Evidence",
        "State": "All",
        "Target_Probabilities": no_evidence_result.values
    })

    # Iteriere über jede Variable, die analysiert werden soll
    for variable in variables_to_analyze:
        original_value = evidence.get(variable)

        # Hole die möglichen Zustände der Variable
        try:
            states = get_variable_states(variable)
        except ValueError as e:
            print(str(e))
            continue

        # Iteriere über alle möglichen Werte der Variable
        for state in states:
            # Setze den neuen Zustand
            evidence[variable] = state

            # Führe Inferenz durch
            prob_dist = inference.query([target_variable], evidence=evidence)

            # Ergebnisse speichern
            results.append({
                "Variable": variable,
                "State": state,
                "Target_Probabilities": prob_dist.values
            })

        # Ursprünglichen Zustand wiederherstellen
        if original_value is not None:
            evidence[variable] = original_value
        else:
            del evidence[variable]

    # Ergebnisse als DataFrame zurückgeben
    return pd.DataFrame(results)


def plot_sensitivity_results(results_df, target_variable="FLOOD_RISK"):
    """
    Plots sensitivity analysis results for each variable as individual bar charts.

    Args:
    - results_df (pd.DataFrame): DataFrame containing 'Variable', 'State', and 'Target_Probabilities' columns.
    - target_variable (str): The name of the target variable being analyzed (default: "FLOOD_RISK").

    Returns:
    None. Displays bar charts for each variable.
    """
    # Extract the probability for 'High' from the Target_Probabilities
    results_df["FLOOD_RISK_High"] = results_df["Target_Probabilities"].apply(lambda x: x[0])

    # Iterate over each unique variable and plot its states and corresponding probabilities
    for variable in results_df["Variable"].unique():
        subset = results_df[results_df["Variable"] == variable]

        # Create the bar plot
        plt.figure(figsize=(8, 6))
        plt.bar(subset["State"], subset["FLOOD_RISK_High"], color="skyblue")

        # Add titles and labels
        plt.title(f"Sensitivity Analysis for {variable} ({target_variable} = High)", fontsize=14)
        plt.xlabel("State", fontsize=12)
        plt.ylabel(f"{target_variable} = High Probability", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.show()


def __get_state_names(variable: str, state_names_dictionary: Dict[str, List[str]],
                      evidence: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Return a subset of the dictionary of state names where keys correspond to the given `variable` of the  Bayesian
    Network and its evidence.

    Parameters
    ----------
    variable : str
        Variable of the Bayesian Network from which the subset of `state_names_dictionary` with the corresponding key is
        taken.
    state_names_dictionary : Dict[str, List[str]]
        Dictionary which keys are variables of the Bayesian Network and which values are the respective discrete states.
    evidence : Optional[List[str]] (default = None)
        Optional list of variables corresponding to the evidence of `variable` in the Bayesian Network.

    Returns
    -------
    Dict[str, List[str]]
        Subset of `state_names_dictionary` where keys correspond to the given `variable` of the Bayesian Network and its
        evidence.
    """

    if evidence is None:
        variable_and_evidence_list = [variable]
    else:
        variable_and_evidence_list = [variable] + evidence

    return dict((v, state_names_dictionary[v]) for v in variable_and_evidence_list if v in state_names_dictionary)


def __get_evidence_card(state_names_dictionary: Dict[str, List[str]],
                        evidence: Optional[List[str]] = None) -> Optional[List[int]]:
    """Return a list containing in each position the cardinality of the discrete states of the corresponding evidence
    variable in the list `evidence`.

    Parameters
    ----------
    state_names_dictionary : Dict[str, List[str]]
        Dictionary which keys are variables of the Bayesian Network and which values are the respective discrete states.
    evidence : Optional[List[str]] (default = None).
        Optional list of variables corresponding to the evidence of `variable` in the Bayesian Network.

    Returns
    -------
    Optional[List[int]]
        A list containing in each position the cardinality of the discrete states of the corresponding evidence variable
        in the list `evidence`. If the list `evidence` is equal to None, None is returned.
    """

    if evidence is None:
        return None
    else:
        return [len(state_names_dictionary[e]) for e in evidence]


def get_tabular_cpd(variable: str, state_names_dictionary: Dict[str, List[str]],
                    values_dictionary: Dict[str, List[List[float]]],
                    evidence_dictionary: Dict[str, List[str]]) -> TabularCPD:
    """Return the Conditional Probability Distribution (CPD) table of a variable according to its parent nodes in the
    Bayesian Network.

    Parameters
    ----------
    variable : str
        Variable of the Bayesian Network for which the CPD table is computed.
    state_names_dictionary : Dict[str, List[str]]
        Dictionary which keys are variables of the Bayesian Network and which values are the respective discrete states.
    values_dictionary : Dict[str, List[List[float]]]
        Dictionary which keys are variables of the Bayesian Network and which values are their respective conditional
        probability distribution values with respect to their evidence.
    evidence_dictionary : Dict[str, List[str]] (default = None).
        Dictionary which keys are variables of the Bayesian Network and which values are their respective evidence.

    Returns
    -------
    TabularCPD
        The CPD table of a variable according to its parent nodes in the Bayesian Network.
    """

    evidence: Optional[List[str]] = evidence_dictionary[variable]

    return TabularCPD(
        variable=variable,
        variable_card=len(state_names_dictionary[variable]),
        values=values_dictionary[variable],
        evidence=evidence,
        evidence_card=__get_evidence_card(state_names_dictionary, evidence),
        state_names=__get_state_names(variable, state_names_dictionary, evidence)
    )


def cpd_to_pandas(cpd: TabularCPD) -> pd.DataFrame:
    """Return the given Conditional Probability Distribution (CPD) table represented as a pandas DataFrame for
    readability.

    Parameters
    ----------
    cpd : TabularCPD
        A CPD table.

    Returns
    -------
    DataFrame
        `cpd` represented as a pandas DataFrame.
    """

    variable = cpd.variable

    evidence = cpd.get_evidence()
    evidence.reverse()

    state_names = cpd.state_names

    if not evidence:
        columns = [['']]
    else:
        columns = pd.MultiIndex.from_product(
            [['{} ({})'.format(e, n) for n in state_names[e]] for e in evidence]
        )

    values = cpd.values

    if values.ndim > 1:
        values = values.reshape(
            values.shape[0], (np.prod(np.array([i for i in values.shape[1:]])))
        )

    return pd.DataFrame(
        values,
        index=['{} ({})'.format(variable, n) for n in state_names[variable]],
        columns=columns
    )


def print_exact_inference(variable: str, infer: VariableElimination,
                          evidence: Optional[Dict[str, List[str]]] = None) -> None:
    """Print the exact inference table of a variable of a Bayesian Network given a specific evidence.

    Parameters
    ----------
    variable : str
        Variable of the Bayesian Network for which the exact inference on its discrete states is computed.
    infer : VariableElimination
        Object to apply exact inference on `variable` with the Variable Elimination method.
    evidence : Optional[Dict[str, List[str]]] (default: None)
        Dictionary which keys are evidence of `variable` in the Bayesian Network and which values are their selected
        state.
    """
    evidence_str = '' if evidence is None else f" | {', '.join([f'{k} = {v}' for k, v in evidence.items()])}"

    print(f"Exact Inference to find P({variable}{evidence_str})\n")
    print(infer.query([variable], show_progress=False, evidence=evidence))


def get_exact_inference_one_state(variable: str, infer: VariableElimination,
                                  evidence: Optional[Dict[str, List[str]]] = None) -> None:
    """Returns the first state of the exact inference table of a variable of a Bayesian Network given a specific evidence.

    Parameters
    ----------
    variable : str
        Variable of the Bayesian Network for which the exact inference on its discrete states is computed.
    infer : VariableElimination
        Object to apply exact inference on `variable` with the Variable Elimination method.
    evidence : Optional[Dict[str, List[str]]] (default: None)
        Dictionary which keys are evidence of `variable` in the Bayesian Network and which values are their selected
        state.
    """
    evidence_str = '' if evidence is None else f" | {', '.join([f'{k} = {v}' for k, v in evidence.items()])}"

    #print(f"Exact Inference to find P({variable}{evidence_str})\n")
    result = infer.query([variable], show_progress=False, evidence=evidence)
    try:
        probability_yes = result.values[0]  # Index 0 entspricht dem Zustand 'Yes'
        #print(f"Wahrscheinlichkeit für {variable} = Yes: {probability_yes}")
        return probability_yes

    except IndexError:
        print(f"Fehler: Zustand 'Yes' für {variable} nicht gefunden oder nicht definiert.")
        return None


def print_approximate_inference(variable: str, infer: ExtendedApproxInference, n_samples=1_000,
                                evidence: Optional[Dict[str, List[str]]] = None, use_weighted_likelihood=False,
                                random_state: int = None) -> None:
    """Print the approximate inference table of a variable of a Bayesian Network given a specific evidence.

    Parameters
    ----------
    variable : str
        Variable of the Bayesian Network for which the approximate inference on its discrete states is computed.
    infer : ExtendedApproxInference
        Object to apply approximate inference on `variable` with the Rejection Sampling or Weighted Likelihood methods.
    n_samples : int (default: 1_000)
        Number of samples to use to compute the approximate inference.
    evidence : Optional[Dict[str, List[str]]] (default: None)
        Dictionary which keys are evidence of `variable` in the Bayesian Network and which values are their selected
        state.
    use_weighted_likelihood : bool (default: False)
        If true, sample by Weighted Likelihood. If false, sample by Rejection Sampling.
    random_state : int (default: None)
        Set a seed for deterministic results on the sampling.
    """
    evidence_str = '' if evidence is None else f" | {', '.join([f'{k} = {v}' for k, v in evidence.items()])}"
    use_weighted_likelihood_str = 'rejection sampling' if not use_weighted_likelihood else 'weighted likelihood'

    print(f"Approximate Inference with {use_weighted_likelihood_str} to find P({variable}{evidence_str})\n")
    print(infer.query(variables=[variable], n_samples=n_samples, show_progress=False, seed=random_state,
                      evidence=evidence, use_weighted_sampling=use_weighted_likelihood))


def apply_discrete_values(variable: str, df: pd.DataFrame, quantiles: List[float],
                          state_names_dictionary: Dict[str, List[str]]) -> None:
    """Discretize the values of a column in a DataFrame according to its discrete state names and given their exact
    inference values.

    Parameters
    ----------
    variable : str
        Column of the DataFrame which its continuous values are discretized.
    df : DataFrame
        pandas Dataframe which column `variable` is discretized.
    quantiles : List[float]
        Quantiles used for the discretization of `variable`.
    state_names_dictionary : Dict[str, List[str]]
        Dictionary which keys are variables of the Bayesian Network and which values are the respective discrete states.
        It is used to select the states in which `variable` is discretized according to `quantiles`.
    """

    state_names = state_names_dictionary[variable]
    state_names = state_names.copy()
    state_names.reverse()

    df[variable] = pd.cut(x=df[variable],
                          bins=quantiles,
                          labels=state_names,
                          include_lowest=True
                          )
