import unittest
import pandas as pd
from loo_encoder.encoder import LeaveOneOutEncoder


class TestLeaveOneOutEncoder(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
                "gender": ["male", "male", "female", "male"],
                "country": ["Germany", "USA", "USA", "UK"],
                "clicks": [10, 33, 47, 21]
            }
        )
        self.y = pd.Series([150, 250, 300, 100], name="orders")

    def tearDown(self):
        pass

    def test_fit_transform_no_weight(self):
        enc = LeaveOneOutEncoder(sigma=0.0, n_smooth=0, cols=['gender', 'country'])
        df_train = enc.fit_transform(self.X, self.y)
        self.assertIsInstance(df_train, pd.DataFrame)
        self.assertEqual(df_train['loo_gender'].values[0], 175)
        self.assertEqual(df_train['loo_country'].values[0], 200)

    def test_fit_transform_with_weight(self):
        enc = LeaveOneOutEncoder(sigma=0.0, n_smooth=0, cols=['gender', 'country'])
        df_train = enc.fit_transform(self.X, self.y, sample_weight=self.X['clicks'])
        self.assertAlmostEqual(df_train['loo_gender'].values[0], 191.666666667)
