"""Test the code in the module util.py.
"""

import math
import unittest
import epicluster as ec


class TestUtilities(unittest.TestCase):

    def test_log_poch(self):
        x = 5
        n = 3
        expected = x * (x+1) * (x+2)
        self.assertAlmostEqual(math.log(expected), ec.log_poch(x, n))

        x = 7.5
        n = 4
        expected = x * (x+1) * (x+2) * (x+3)
        self.assertAlmostEqual(math.log(expected), ec.log_poch(x, n))

        self.assertEqual(0, ec.log_poch(x, 0))


if __name__ == '__main__':
    unittest.main()
