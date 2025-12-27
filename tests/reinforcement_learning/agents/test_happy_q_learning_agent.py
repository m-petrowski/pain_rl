from unittest.mock import ANY
import json

from src.reinforcement_learning.agents.happy_q_learning_agent import HappyQLearningAgent
from src.reinforcement_learning.environments.pain_markov_chain import PainMarkovChain


def test_update_q_with_positive_reward(mocker):
    state = (0, 0)
    next_state = (0, 1)
    obj_reward = 1
    env_mock = mocker.Mock()
    env_mock.get_player_position.return_value = state
    env_mock.step.return_value = (next_state, obj_reward)
    pain_model = PainMarkovChain()
    agent = HappyQLearningAgent(env_mock, 1, 0, 0, 0, pain_model)

    old_state_value = agent.get_value_at_position(state)
    agent.act_one_step()
    new_state_value = agent.get_value_at_position(state)

    env_mock.get_player_position.assert_called_once()
    env_mock.step.assert_called_with(ANY)

    assert old_state_value < new_state_value

def test_from_param_map():
    with open("resources/param_map.json") as file:
        param_map = json.load(file)

    agent =HappyQLearningAgent.from_param_map(param_map)
    print("done")