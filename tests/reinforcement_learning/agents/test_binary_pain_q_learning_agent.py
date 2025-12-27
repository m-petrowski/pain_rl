import json
import random

from src.reinforcement_learning.agents.binary_pain_q_learning_agent import BinaryPainQLearningAgent
from src.reinforcement_learning.agents.happy_q_learning_agent import HappyQLearningAgent

import warnings
warnings.simplefilter("error", RuntimeWarning)

def test_binary_and_happy_agent():
    """
    BinaryPainQLearningAgent and HappyQLearningAgent produce the same result when w4 = 0.
    Params including random seed should be the same
    """
    with open("resources/param_map_binary_pain_is_0.json") as file1:
        param_map_hmm = json.load(file1)

    with open("resources/param_map_markov_chain_pain_is_0.json") as file2:
        param_map_chain = json.load(file2)

    random.seed(50)
    agent_hmm = BinaryPainQLearningAgent.from_param_map(param_map_hmm)
    random.seed(50)
    agent_chain = HappyQLearningAgent.from_param_map(param_map_chain)
    assert agent_hmm.q == agent_chain.q

    for step in range(1, 1001):
        random.seed(50 + step)
        agent_hmm.act_one_step()
        random.seed(50 + step)
        agent_chain.act_one_step()

        assert agent_hmm.q == agent_chain.q





