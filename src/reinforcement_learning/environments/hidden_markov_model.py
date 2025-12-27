from enum import Enum
from typing import List

import numpy as np

class HiddenMarkovModel:
    """
    Hidden Markov Model with implemented Forward Algorithm
    Source: https://en.wikipedia.org/wiki/Forward_algorithm
    """
    def __init__(self, hidden_states, observation_states, transition_probabilities: List[List[float]], emission_probabilities: List[List[float]], prior_probabilities: List[float]):
        self.hidden_states = hidden_states
        self.hidden_states_to_index = dict(zip(self.hidden_states, range(len(self.hidden_states))))
        self.observation_states = observation_states
        self.observation_states_to_index = dict(zip(self.observation_states, range(len(self.observation_states))))


        self.transition_probabilities = np.array(transition_probabilities)
        assert self.transition_probabilities.shape[0] == self.transition_probabilities.shape[1]

        self.emission_probabilities = np.array(emission_probabilities)
        assert self.emission_probabilities.shape[0] == self.emission_probabilities.shape[1]

        self.prior_probabilities = np.array(prior_probabilities) #liegender vector

        self.current_step = 0
        self.scaled_alphas = []
        self.scaling_factors = []

    def step(self, observation):
        """Returns maximum a posteriori estimate after the step"""
        assert observation in self.observation_states
        observation_index = self.observation_states_to_index[observation]

        b_t = self.emission_probabilities[observation_index]

        if self.current_step == 0:
            alpha = b_t * self.prior_probabilities
        else:
            alpha = b_t * (self.transition_probabilities @ self.scaled_alphas[self.current_step - 1])

        scaling_factor = np.sum(alpha)
        scaled_alpha = alpha / scaling_factor

        self.scaled_alphas.append(scaled_alpha)
        self.scaling_factors.append(scaling_factor)
        self.current_step += 1

        return self.get_maximum_a_posteriori_estimate()

    def get_maximum_a_posteriori_estimate(self):
        """Returns the most probable current hidden state (MAP estimate)."""
        scaled_alpha = self.scaled_alphas[-1]
        idx = int(np.argmax(scaled_alpha))
        return self.hidden_states[idx]

    def get_latest_scaled_alpha(self):
        """Returns the current unnormalized belief state as a list."""
        return self.scaled_alphas[self.current_step - 1].tolist()

    def get_latest_scaling_factor(self):
        return self.scaling_factors[-1]

    def get_latest_belief_states(self):
        """Returns the normalized belief state as a list of floats."""
        scaled_alpha = self.scaled_alphas[-1]
        total = np.sum(scaled_alpha)
        if total == 0:
            return [1.0 / len(scaled_alpha)] * len(scaled_alpha)
        belief_states = (scaled_alpha / total)
        assert np.isclose(np.sum(belief_states), 1.0)
        return belief_states.tolist()

    def get_belief_of_hidden_state(self, state):
        """Returns the belief (posterior probability) of a specific hidden state."""
        return self.get_latest_belief_states()[self.hidden_states_to_index[state]]

class PainStates(Enum):
    NO_PAIN = 0
    PAIN = 1

class PainObservations(Enum):
    HARMLESS = 0
    NOXIOUS = 1

class BinaryPainHiddenMarkovModel(HiddenMarkovModel):
    def __init__(self, transition_probabilities: List[List[float]], emission_probabilities: List[List[float]], prior_probabilities: List[float]):
        super().__init__([PainStates.NO_PAIN, PainStates.PAIN], [PainObservations.HARMLESS, PainObservations.NOXIOUS], transition_probabilities, emission_probabilities,
                         prior_probabilities)

    def get_current_pain_probability(self):
        if self.current_step == 0:
            return 0
        else:
            return super().get_belief_of_hidden_state(PainStates.PAIN)


