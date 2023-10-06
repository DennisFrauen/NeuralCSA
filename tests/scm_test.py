import unittest
from data.scms import SCM_binary
import utils.utils as utils
import numpy as np

class TestSCMBinary(unittest.TestCase):
    def test_marginalization(self):
        # Test that the observed propensity score is the marginal of the full propensity score
        config = utils.load_yaml("/experiments/sim_binary/config")
        scm = SCM_binary(config["data"])
        x_fix = np.random.uniform(-1, 1, (1, 1))
        p_u1_x = scm.get_pu_given_x(x_fix, 1)
        p_a1_xu1 = scm.propensity_full(x_fix, 1, 1)
        p_a1_xu0 = scm.propensity_full(x_fix, 0, 1)
        p_a1_x = scm.propensity_obs(x_fix, 1)
        test_marginal = p_a1_xu0 * (1 - p_u1_x) + p_a1_xu1 * p_u1_x
        self.assertEqual(test_marginal, p_a1_x,
                         "Observed propensity score should be marginal of full propensity score")

    def test_confounding_strength(self):
        # Test that the confounding strength is equal to gamma
        config = utils.load_yaml("/experiments/sim_binary/config")
        scm = SCM_binary(config["data"])
        x_fix = np.random.uniform(-1, 1, (1, 1))
        #Density ratios
        p_a1_xu1 = scm.propensity_full(x_fix, 1, 1)
        p_a1_xu0 = scm.propensity_full(x_fix, 0, 1)
        p_a1_x = scm.propensity_obs(x_fix, 1)
        ratio1 = p_a1_xu1 / p_a1_x
        ratio0 = p_a1_xu0 / p_a1_x
        #Bounds
        splus = scm.get_splus(x_fix, 1)
        sminus = scm.get_sminus(x_fix, 1)
        condition1 = ratio1 == splus and ratio0 == sminus
        condition2 = ratio1 == sminus and ratio0 == splus
        result = condition1 or condition2

        self.assertTrue(result, "ratio bounds test failed")


if __name__ == '__main__':
    unittest.main()