"""Test class for Markov model."""
from __future__ import division
import unittest
from scipy.io import loadmat
from markov import *


class TestHomework5(unittest.TestCase):
    """Tests for MMs"""
    def test_mm_train(self):
        cycle_data = ['a b c d',
                      'b c d a',
                      'c d a b',
                      'd a b c']

        model = train_markov_chain(cycle_data)

        for token in ['a', 'b', 'c', 'd']:
            assert np.allclose(model['prior'][token], 0.25), "Starting prior was not uniform over the four possible " \
                                                             "tokens."

        cycle_order = ['a', 'b', 'c', 'd', 'a']

        for i in range(4):
            probs = model['transitions'][cycle_order[i]]

            for next_word in ['a', 'b', 'c', 'd']:
                if next_word == cycle_order[i + 1]:
                    assert np.allclose(probs[next_word], 3 / 4), "Deterministic transition probability to " \
                                                                        "correct next letter was wrong"
                else:
                    assert next_word not in probs, "Transition probability to an incorrect next letter was in the " \
                                                   "dictionary. You should not compute or store the transitions that " \
                                                   "never occur in the training data, or else you lose the benefit of" \
                                                   "the sparsity of the data and computation on real data will be " \
                                                   "too expensive."
                assert np.allclose(probs['\n'], 1 / 4), "Probability of stop token is not correct."

        skewed_data = ['a a a b a']

        model = train_markov_chain(skewed_data)

        assert np.allclose(model['prior']['a'], 1.0), "Starting prior was not correct for non-uniform data"

        assert np.allclose(model['transitions']['a']['a'], 2 / 4), "Transition probability appears incorrect on " \
                                                                   "non-uniform data."
        assert np.allclose(model['transitions']['a']['b'], 1 / 4), "Transition probability appears incorrect on " \
                                                                   "non-uniform data."
        assert np.allclose(model['transitions']['a']['\n'], 1 / 4), "Transition probability appears incorrect on " \
                                                                    "non-uniform data (for the probability of the " \
                                                                    "stop token)."
        assert 'b' not in model['transitions']['b'], "Transition probability has an entry for b-b, when that " \
                                                     "transition does not occur in the data."
        assert np.allclose(model['transitions']['b']['a'], 1), "Transition probability appears incorrect on " \
                                                               "non-uniform data."

    def test_mm_sample(self):
        # test some corner cases for sampling
        deterministic_model = {'prior': {'a': 1.0},
                               'transitions': {'a': {'b': 1.0},
                                               'b': {'c': 1.0},
                                               'c': {'d': 1.0},
                                               'd': {'\n': 1.0}}}

        sequences = sample_markov_chain(deterministic_model, 1)

        assert sequences[0] == ['a', 'b', 'c', 'd', '\n'], "Deterministic model generated incorrect sequence."

        # test that the method handles multi-character strings correctly
        string_model = {'prior': {'foo': 1.0},
                        'transitions': {'foo': {'bar': 1.0},
                                        'bar': {'foo': 1.0}}}

        sequences = sample_markov_chain(string_model, 1, max_length=10)

        assert "".join(sequences[0]) == 'foobarfoobarfoobarfoobarfoobar\n', \
            "Sampler does not successfully generate string sequences. Could be an issue with appending strings as " \
            "nonatomic iterables."

if __name__ == '__main__':
    unittest.main()
