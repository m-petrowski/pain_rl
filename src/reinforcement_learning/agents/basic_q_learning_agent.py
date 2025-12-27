from collections import defaultdict
import random
import importlib
from typing import Tuple

from src.reinforcement_learning.agents.visualizable_agent import VisualizableAgent
from src.reinforcement_learning.environments.grid_world import BasicActions, GridWorld


def default_q_values():
    return defaultdict(float)

def default_q_table():
    return defaultdict(default_q_values)

class BasicQLearningAgent(VisualizableAgent):
    def __init__(self, environment: GridWorld, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q = default_q_table()
        self.steps_completed = 0

    @classmethod
    def from_param_map(cls, param_map: dict):
        agent_params = param_map["agent_params"]
        environment_params = param_map["environment_params"]
        module_path, environment_class = param_map["environment_class"].rsplit('.', 1)
        module = importlib.import_module(module_path)
        environment_class = getattr(module, environment_class)
        environment = environment_class.from_param_map(environment_params)
        return cls(environment, **agent_params)

    def _choose_action(self, state: Tuple[int, int], epsilon: float) -> BasicActions:
        """
        Implementation of the epsilon-greedy-policy. Returns the action that is chosen. If epsilon = 0 it returns just the greedy action
        """
        assert isinstance(state, tuple) and isinstance(state[0], int) and isinstance(state[1], int)
        if random.random() >= epsilon:
            action = self._get_max_q_action(state)
        else:
            action = random.choice(list(BasicActions))

        return action

    def _get_max_q_action(self, state: Tuple[int, int]) -> BasicActions:
        return self.get_max_q_action_value_pair(state)[0]

    def _get_max_q_value(self, state: Tuple[int, int]) -> float:
        return self.get_max_q_action_value_pair(state)[1]

    def get_max_q_action_value_pair(self, state: Tuple[int, int]) -> tuple[BasicActions, float]:
        tuples = [(action, self.q[state][action]) for action in list(BasicActions)]
        random.shuffle(tuples)
        pair = max(tuples, key=lambda x: x[1])
        return pair

    def choose_action_epsilon_greedy(self, state: Tuple[int, int]) -> BasicActions:
        """Uses the epsilon of the instance"""
        return self._choose_action(state, self.epsilon)

    def update_q(self, state: Tuple[int, int], action: BasicActions, next_state: Tuple[int, int], subjective_reward: float):
        self.q[state][action] = self.q[state][action] + self.alpha * (subjective_reward + self.gamma * self._get_max_q_value(next_state) - self.q[state][action])

    def calculate_subjective_reward(self, state: Tuple[int, int], action: BasicActions, objective_reward: float):
        return objective_reward

    def act_one_step(self) -> Tuple[Tuple[int, int], BasicActions, Tuple[int, int], float, float]:
        state = self.environment.get_player_position()
        action = self. choose_action_epsilon_greedy(state)
        next_state, objective_reward = self.environment.step(action)
        subjective_reward = self.calculate_subjective_reward(state, action, objective_reward)
        self.update_q(state, action, next_state, subjective_reward)
        self.steps_completed += 1
        return state, action, next_state, objective_reward, subjective_reward

    def get_agent_position(self) -> Tuple[int, int]:
        return self.environment.get_player_position()

    def get_value_at_position(self, state: Tuple[int, int]) -> float:
        return self._get_max_q_value(state)

    def get_cell_visit_count(self, pos: Tuple[int, int]) -> int:
        return self.environment.get_cell_visit_count(pos)