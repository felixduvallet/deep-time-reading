import unittest
import numpy as np
from clock_reading.clock_evaluation import compute_time_errors


class TestCase(unittest.TestCase):

    def test_correct_time(self):
        pred = [(5, 10)]
        true = [(5, 10)]
        errs = compute_time_errors(pred, true)
        ref = np.array([[0, 0, 0]])
        np.testing.assert_equal(ref, errs)

    def test_off_one_hour(self):
        pred = [(5, 10)]
        true = [(6, 10)]
        errs = compute_time_errors(pred, true)
        ref = np.array([[60, 1, 0]])  # [total (in minutes), hours, minutes]
        np.testing.assert_equal(ref, errs)

    def test_off_one_hour_symmetric(self):
        pred = [(5, 10)]
        true = [(6, 10)]
        errs = compute_time_errors(true, pred)
        ref = np.array([[60, 1, 0]])
        np.testing.assert_equal(ref, errs)

    def test_off_one_minute(self):
        pred = [(1, 59)]
        true = [(2, 1)]
        errs = compute_time_errors(pred, true)
        ref = np.array([[2, 1, 2]])  # Note hours are treated independently.
        np.testing.assert_equal(ref, errs)

    def test_wraparound(self):
        pred = [(11, 59)]
        true = [(0, 0)]
        errs = compute_time_errors(pred, true)
        ref = np.array([[1, 1, 1]])
        np.testing.assert_equal(ref, errs)

        errs = compute_time_errors(true, pred)  # flip order.
        np.testing.assert_equal(ref, errs)





if __name__ == '__main__':
    unittest.main()
