from unittest.mock import Mock

from src.reinforcement_learning.environments.pain_markov_chain import PainMarkovChain


def test_pain_states_sorted():
    input_states = (1, 0.5, 0, 0.75)
    pain_model = PainMarkovChain(pain_states=input_states)
    pain_states = pain_model.pain_states

    assert sorted(input_states) == pain_states

def test_pain_moves_to_max_pain():
    mock_rng = Mock()
    mock_rng.random.return_value = 0.001
    pain_model = PainMarkovChain(pain_states = (0, 0.25, 0.5, 0.75, 1), pain_probability = 0.01, random_number_generator=mock_rng)
    pain_model.step()
    assert pain_model.get_pain() == 1
    
def test_cannot_exceed_max_pain():
    mock_rng = Mock()
    mock_rng.random.return_value = 0.001
    pain_model = PainMarkovChain(pain_states=(0, 0.25, 0.5, 0.75, 1), pain_probability=0.01, random_number_generator=mock_rng)
    pain_model.step()
    pain_model.step()
    assert pain_model.get_pain() == 1

def test_cannot_go_below_min_pain(mocker):
    mock_rng = Mock()
    mock_rng.random.return_value = 0.9
    pain_model = PainMarkovChain(pain_states=(0, 0.25, 0.5, 0.75, 1), pain_probability=0.01, random_number_generator=mock_rng)
    pain_model.step()
    pain_model.step()
    assert pain_model.get_pain() == 0

def test_decreases_pain_correctly():
    mock_rng = Mock()
    mock_rng.random.return_value = 0.001
    pain_model = PainMarkovChain(pain_states=(0, 0.25, 0.5, 0.75, 1), pain_probability=0.01, random_number_generator=mock_rng)
    pain_model.step()
    mock_rng.random.return_value = 0.9
    pain_model.step()
    assert pain_model.get_pain() == 0.75
