import random

class PainMarkovChain:
    def __init__(self, pain_states = (0, 0.25, 0.5, 0.75, 1), pain_probability = 0.01, seed=None, random_number_generator=None):
        assert len(pain_states) >= 1
        self.pain_states = sorted(pain_states)
        self.num_pain_states = len(pain_states)
        self.pain_probability = pain_probability
        self.current_pain_index = 0
        self.random = random_number_generator or random.Random(seed)

    def step(self):
        if self.random.random() < self.pain_probability:
            self.current_pain_index = self.num_pain_states - 1
        elif self.current_pain_index > 0:
            self.current_pain_index -= 1
        else:
            self.current_pain_index = 0

    def get_pain(self):
        return self.pain_states[self.current_pain_index]

    def get_max_pain(self):
        return self.pain_states[self.num_pain_states - 1]

