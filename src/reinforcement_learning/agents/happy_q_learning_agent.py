from collections import defaultdict
import importlib
from typing import Tuple

from src.reinforcement_learning.agents.basic_q_learning_agent import BasicQLearningAgent
from src.reinforcement_learning.agents.visualizable_agent import PainVisualizationAgent
from src.reinforcement_learning.environments.grid_world import BasicActions, GridWorld
from src.reinforcement_learning.environments.pain_markov_chain import PainMarkovChain


#Cannot use lambdas because object must be pickable
def default_q_values():
    return defaultdict(float)

def default_q_table():
    return defaultdict(default_q_values)


class HappyQLearningAgent(BasicQLearningAgent, PainVisualizationAgent):
    def __init__(self, environment: GridWorld, w1: float, w2: float, w3: float, w4: float, pain_model: PainMarkovChain, aspiration_level=0.5, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(environment, alpha=alpha, gamma=gamma, epsilon=epsilon)

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

        self.pain_model = pain_model

        self.aspiration_level = aspiration_level

        self.q = default_q_table()

    @classmethod
    def from_param_map(cls, param_map: dict):
        agent_params = param_map["agent_params"]
        agent_params["pain_model"] = PainMarkovChain(**agent_params["pain_model"])
        environment_params = param_map["environment_params"]
        module_path, environment_class = param_map["environment_class"].rsplit('.', 1)
        module = importlib.import_module(module_path)
        environment_class = getattr(module, environment_class)
        environment = environment_class.from_param_map(environment_params)
        return cls(environment, **agent_params)

    def calculate_subjective_reward(self, state: Tuple[int, int], action: BasicActions, objective_reward: float) -> float:
        self.pain_model.step()
        return self.w1 * objective_reward + self.w2 * (objective_reward - self.q[state][action]) + self.w3 * (objective_reward - self.aspiration_level) - self.w4 * self.pain_model.get_pain()

    def get_subjective_pain(self) -> float:
        return self.w4 * self.pain_model.get_pain()

    def get_max_pain(self):
        return self.pain_model.get_max_pain()

    def get_current_pain(self):
        return self.pain_model.get_pain()
