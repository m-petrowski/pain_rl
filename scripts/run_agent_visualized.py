import random
random.seed(600)


from src.reinforcement_learning.agents.binary_pain_q_learning_agent import BinaryPainQLearningAgent
from src.reinforcement_learning.environments.grid_world import GridWorld
from src.reinforcement_learning.environments.hidden_markov_model import BinaryPainHiddenMarkovModel
from src.visualization.agent_visualizer import AgentLearningVisualizer
from experiments.pain_models import NORMAL_PAIN_PROBABILITIES,CHRONIC_PAIN_PROBABILITIES


# Runs a visual representation of the agent in the grid world

grid_size = 7
possible_starting_positions = [(0,6), (1, 6), (0, 5), (1, 5)]
obstacle_positions = [(3, 2), (3, 3), (3, 4), (2, 3), (4, 3), (3,0), (3, 6)]
possible_initial_food_positions = [(5, 0), (6, 0), (5, 1), (6, 1)]

grid_world = GridWorld(grid_size, possible_starting_positions, possible_initial_food_positions, obstacle_positions, lifetime_learning=True, stationary=True)

normal_transition_probabilities = NORMAL_PAIN_PROBABILITIES["transition_probabilities"]
normal_emission_probabilities = NORMAL_PAIN_PROBABILITIES["emission_probabilities"]
normal_prior_probabilities = NORMAL_PAIN_PROBABILITIES["prior_probabilities"]
normal_pain_model = BinaryPainHiddenMarkovModel(normal_transition_probabilities, normal_emission_probabilities, normal_prior_probabilities)

chronic_transition_probabilities = CHRONIC_PAIN_PROBABILITIES["transition_probabilities"]
chronic_emission_probabilities = CHRONIC_PAIN_PROBABILITIES["emission_probabilities"]
chronic_prior_probabilities = CHRONIC_PAIN_PROBABILITIES["prior_probabilities"]
chronic_pain_model = BinaryPainHiddenMarkovModel(chronic_transition_probabilities, chronic_emission_probabilities, chronic_prior_probabilities)
normal_agent = BinaryPainQLearningAgent(grid_world, 0.1, 0.1, 0, 0.5, normal_pain_model, aspiration_level=0, alpha=0.7, gamma=0.99, epsilon=0.1)
chronic_agent = BinaryPainQLearningAgent(grid_world, 0.1, 0.3, 0, 0.5, chronic_pain_model, aspiration_level=0, alpha=0.3, gamma=0.99, epsilon=0.1)


visualizer = AgentLearningVisualizer(normal_agent, step_interval=10, draw_policy_and_values=True, draw_visit_counts=False)
visualizer.start()