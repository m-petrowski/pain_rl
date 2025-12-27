import numpy as np
import random
import math

from src.reinforcement_learning.environments.hidden_markov_model import HiddenMarkovModel, BinaryPainHiddenMarkovModel, \
    PainObservations

pain_transition_probabilities = [
    [0.3, 0.2],
    [0.7, 0.8]
]

pain_emission_probabilities = [
    [0.4, 0.4],
    [0.6, 0.6]
]

pain_prior_probabilities = [0.223, 0.777]

def test_BinaryPainHMM_and_HMM_have_same_results():
    hmm_model = HiddenMarkovModel(["no_pain", "pain"], ["harmless", "noxious"], pain_transition_probabilities,
                                   pain_emission_probabilities, pain_prior_probabilities)
    bphmm_model = BinaryPainHiddenMarkovModel(pain_transition_probabilities,
                                   pain_emission_probabilities, pain_prior_probabilities)
    hmm_model.step("harmless")
    bphmm_model.step(PainObservations.HARMLESS)
    hmm_alpha = hmm_model.get_latest_scaled_alpha()
    bphmm_alpha = bphmm_model.get_latest_scaled_alpha()
    hmm_belief = hmm_model.get_latest_belief_states()
    bphmm_belief = bphmm_model.get_latest_belief_states()

    assert np.allclose(hmm_alpha, bphmm_alpha)
    assert np.allclose(hmm_belief, bphmm_belief)

def test_belief_state_sums_to_one():
    hmm_model = HiddenMarkovModel(["no_pain", "pain"], ["harmless", "noxious"], pain_transition_probabilities,
                                  pain_emission_probabilities, pain_prior_probabilities)
    for i in range(100):
        observation = random.choice(["harmless", "noxious"])
        hmm_model.step(observation)
        result = sum(hmm_model.get_latest_belief_states())
        assert math.isclose(result, 1.0)

transition_probabilities = [
    [0.5, 0.3],
    [0.5, 0.7]
]

emission_probabilities = [
    [0.8, 0.4],
    [0.2, 0.6]
]

prior_probabilities = [0.375, 0.625]

#Example and results from https://www.youtube.com/watch?v=9-sPm4CfcD0
#Sequence: sad, sad, happy
def test_alpha_after_one_step():
    pain_model = HiddenMarkovModel(["rainy", "sunny"], ["sad", "happy"], transition_probabilities, emission_probabilities, prior_probabilities)
    pain_model.step("sad")
    result = pain_model.get_latest_scaled_alpha()
    scaling_factor = pain_model.get_latest_scaling_factor()

    assert np.allclose(result, [0.3 / scaling_factor, 0.25 / scaling_factor])

def test_alpha_after_two_steps():
    pain_model = HiddenMarkovModel(["rainy", "sunny"], ["sad", "happy"], transition_probabilities,
                                   emission_probabilities, prior_probabilities)
    pain_model.step("sad")
    scaling_factor1 = pain_model.get_latest_scaling_factor()
    pain_model.step("sad")
    scaling_factor2 = pain_model.get_latest_scaling_factor()
    result = pain_model.get_latest_scaled_alpha()

    assert np.allclose(result, [0.18 / scaling_factor1 / scaling_factor2, 0.13 / scaling_factor1 / scaling_factor2])

def test_alpha_after_three_steps():
    pain_model = HiddenMarkovModel(["rainy", "sunny"], ["sad", "happy"], transition_probabilities,
                                   emission_probabilities, prior_probabilities)
    pain_model.step("sad")
    scaling_factor1 = pain_model.get_latest_scaling_factor()
    pain_model.step("sad")
    scaling_factor2 = pain_model.get_latest_scaling_factor()
    pain_model.step("happy")
    scaling_factor3 = pain_model.get_latest_scaling_factor()
    result = pain_model.get_latest_scaled_alpha()

    assert np.allclose(result, [0.0258 / scaling_factor1 / scaling_factor2 / scaling_factor3, 0.1086 / scaling_factor1 / scaling_factor2 / scaling_factor3])

def number_to_pain_observation(x: int):
    if x == 0:
        return PainObservations.HARMLESS
    else:
        return PainObservations.NOXIOUS

def test_normal_pain_belief_state_against_pomgranate():
    from pomegranate.hmm import DenseHMM
    from pomegranate.distributions import Categorical

    no_pain_dist = Categorical([[0.9, 0.1]])
    pain_dist = Categorical([[0.2, 0.8]])

    model = DenseHMM()
    model.add_distributions([no_pain_dist, pain_dist])
    model.add_edge(model.start, no_pain_dist, 0.777)
    model.add_edge(model.start, pain_dist, 0.223)
    model.add_edge(no_pain_dist, no_pain_dist, 0.8)
    model.add_edge(no_pain_dist, pain_dist, 0.2)
    model.add_edge(pain_dist, no_pain_dist, 0.7)
    model.add_edge(pain_dist, pain_dist, 0.3)

    observations = [random.choice([0, 1]) for _ in range(100)]
    
    X = np.array([[[x] for x in observations]], dtype=int)

    log_probs = model.forward(X)[0]  # shape: (T, N)
    probs = np.exp(log_probs)
    normalized_probs = probs / probs.sum(axis=1, keepdims=True)

    # Compare with custom HMM forward algorithm
    bphmm_model = BinaryPainHiddenMarkovModel(
        [[0.8, 0.7], [0.2, 0.3]],  # transposed transition matrix
        [[0.9, 0.2], [0.1, 0.8]],  # emission matrix
        [0.777, 0.223]  # prior
    )

    belief_states_custom = []
    pain_observations = map(number_to_pain_observation, observations)
    for obs in pain_observations:
        bphmm_model.step(obs)
        belief_states_custom.append(bphmm_model.get_latest_belief_states())
    belief_states_custom = np.array(belief_states_custom)

    assert np.allclose(belief_states_custom, normalized_probs, atol=1e-5)


def test_chronic_pain_belief_state_against_pomgranate():
    from pomegranate.hmm import DenseHMM
    from pomegranate.distributions import Categorical

    no_pain_dist = Categorical([[0.4, 0.6]])
    pain_dist = Categorical([[0.4, 0.6]])

    model = DenseHMM()
    model.add_distributions([no_pain_dist, pain_dist])
    model.add_edge(model.start, no_pain_dist, 0.223)
    model.add_edge(model.start, pain_dist, 0.777)
    model.add_edge(no_pain_dist, no_pain_dist, 0.3)
    model.add_edge(no_pain_dist, pain_dist, 0.7)
    model.add_edge(pain_dist, no_pain_dist, 0.2)
    model.add_edge(pain_dist, pain_dist, 0.8)

    observations = [random.choice([0, 1]) for _ in range(100)]

    X = np.array([[[x] for x in observations]], dtype=int)

    log_probs = model.forward(X)[0]  # shape: (T, N)
    probs = np.exp(log_probs)
    normalized_probs = probs / probs.sum(axis=1, keepdims=True)

    # Compare with custom HMM forward algorithm
    bphmm_model = BinaryPainHiddenMarkovModel(
        [[0.3, 0.2], [0.7, 0.8]],  # transposed transition matrix
        [[0.4, 0.4], [0.6, 0.6]],  # emission matrix
        [0.223, 0.777]  # prior
    )

    belief_states_custom = []
    pain_observations = map(number_to_pain_observation, observations)
    for obs in pain_observations:
        bphmm_model.step(obs)
        belief_states_custom.append(bphmm_model.get_latest_belief_states())
    belief_states_custom = np.array(belief_states_custom)

    assert np.allclose(belief_states_custom, normalized_probs, atol=1e-5)


