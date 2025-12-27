import pytest
from unittest.mock import MagicMock
from src.reinforcement_learning.agents.basic_q_learning_agent import BasicQLearningAgent
from src.reinforcement_learning.environments.grid_world import BasicActions, GridWorld

@pytest.fixture
def mock_environment():
    env = MagicMock(spec=GridWorld)
    env.get_player_position.return_value = (0, 6)
    env.step.return_value = ((1, 6), 1)
    env.get_cell_visit_count.return_value = 5
    return env

@pytest.fixture
def agent(mock_environment):
    return BasicQLearningAgent(mock_environment, alpha=0.1, gamma=0.99, epsilon=0.1)

def test_update_q(agent):
    state = (0, 6)
    action = BasicActions.UP
    next_state = (0, 5)
    subjective_reward = 10
    agent.update_q(state, action, next_state, subjective_reward)
    assert agent.q[state][action] > 0

def test_act_one_step(agent):
    result = agent.act_one_step()
    assert len(result) == 5
    assert result[0] == (0, 6)  #Initial state
    assert result[1] in list(BasicActions)  #Action
    assert result[2] == (1, 6)  #Next state
    assert result[3] == 1  #Objective reward
    assert result[4] > 0  #Subjective reward

def test_get_agent_position(agent):
    position = agent.get_agent_position()
    assert position == (0, 6)

def test_get_value_at_position(agent):
    state = (0, 6)
    action = BasicActions.UP
    next_state = (0, 5)
    subjective_reward = 10
    agent.update_q(state, action, next_state, subjective_reward)
    value = agent.get_value_at_position(state)
    assert value == 1.0