import pandas as pd
import numpy as np
from loo_encoder.utils import convert_input, get_obj_cols, check_random_state


class LeaveOneOutEncoder(object):
    """Leave one out coding for categorical features.

        Parameters
        ----------

        cols: list
            a list of columns to encode, if None, all string columns will be encoded
        return_weight_feature: bool
            if True, the sum of weights will be returned with prefix "cnt_"
        n_smooth: integer
            number of weights added to each data point, defaults to 0
        handle_unknown: str
            options are 'ignore' and 'impute', defaults to 'impute', which will impute the category -1.
        random_state: integer or a np.random.RandomState instance
        sigma : float, Standard deviation (spread or "width") of the distribution.

        Example
        -------
        >>>import pandas as pd
        >>>from loo_encoder.encoder import LeaveOneOutEncoder
        >>>enc = LeaveOneOutEncoder(cols=['gender', 'country'], handle_unknown='impute', sigma=0.02, random_state=42)
        >>>X = pd.DataFrame(
        >>>    {
        >>>        "gender": ["male", "male", "female", "male"],
        >>>        "country": ["Germany", "USA", "USA", "UK"],
        >>>        "clicks": [10, 33, 47, 21]
        >>>    }
        >>>)
        >>>y = pd.Series([150, 250, 300, 100], name="orders")
        >>>df_train = enc.fit_transform(X=X, y=y, sample_weight=X['clicks'])
        >>>print(df_train.info())
    """

    def __init__(self, cols=None, return_weight_feature=False, handle_unknown='impute',
                 random_state=None, sigma=0.05, n_smooth=0):
        self.return_weight_feature = return_weight_feature
        self.cols = cols
        self.mapping = {}
        self.handle_unknown = handle_unknown
        self._mean = None
        self.random_state_ = check_random_state(random_state)
        self.sigma = sigma
        self.n_smooth = n_smooth

    def fit(self, X, y, **kwargs):
        """Fit encoder according to X and y.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------

        self : encoder
            Returns self.
        """

        # first check the type and shape
        X = convert_input(X)
        y = pd.Series(y, name='_target_')
        assert X.shape[0] == y.shape[0]

        df_train = self._create_df_train(X=X, y=y, sample_weight=kwargs.get("sample_weight", None))
        self._leave_one_out_encoding(df=df_train)

    def fit_transform(self, X, y, **kwargs):
        # first check the type and shape
        X = convert_input(X)
        y = pd.Series(y, name='_target_')
        assert X.shape[0] == y.shape[0]

        df_train = self._create_df_train(X=X, y=y, sample_weight=kwargs.get("sample_weight", None))
        self._leave_one_out_encoding(df=df_train)
        df_train = self._transform_train(df_train=df_train)
        return self._output_encoded(df_train)

    def transform(self, X, **kwargs):
        X = convert_input(X)
        df_test = self._transform_test(df_test=X)
        return self._output_encoded(df_test)

    def _output_encoded(self, df):
        cols = [c for c in ['_weighted_target_', '_weight_', '_target_'] if c in df.index]
        return df.drop(cols, axis=1)

    def _leave_one_out_encoding(self, df):
        self._mean = df['_weighted_target_'].sum() / df['_weight_'].sum()
        for feature in self.cols:
            df_group = df.groupby(feature, as_index=False).agg({
                '_weighted_target_': 'sum',
                '_weight_': 'sum'
            })
            df_group.rename(columns={
                '_weighted_target_': 'group_sum',
                '_weight_': 'group_weight_sum'
            }, inplace=True)
            df_group.group_sum += self.n_smooth * self._mean
            df_group['group_weight_sum'] += self.n_smooth
            self.mapping[feature] = df_group

    def _transform_train(self, df_train):
        for feature in self.cols:
            df_train = df_train.merge(self.mapping[feature], on=feature)
            df_train['loo_' + feature] = (df_train.group_sum - df_train['_weighted_target_']) / (
                    df_train.group_weight_sum - df_train._weight_)
            df_train['loo_' + feature].fillna(self._mean, inplace=True)
            df_train['cnt_' + feature] = df_train.group_weight_sum - df_train._weight_
            df_train['loo_' + feature] *= self.random_state_.normal(1, self.sigma, df_train.shape[0])
            df_train.drop(['group_weight_sum', 'group_sum'], axis=1, inplace=True)
        return df_train

    def _transform_test(self, df_test):
        for feature in self.cols:
            df_test = df_test.merge(self.mapping[feature], left_on=feature, right_on=feature, how='left',
                                    left_index=False, right_index=False)
            df_test['loo_' + feature] = df_test.group_sum / df_test.group_weight_sum
            df_test['cnt_' + feature] = df_test.group_weight_sum
            if self.handle_unknown == 'impute':
                df_test['loo_' + feature].fillna(self._mean, inplace=True)
                df_test['cnt_' + feature].fillna(df_test.group_weight_sum.mean(skipna=True), inplace=True)
            df_test.drop(['group_weight_sum', 'group_sum'], axis=1, inplace=True)
        return df_test

    def _create_df_train(self, X, y, sample_weight=None):
        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)

        if sample_weight is None:
            weight = pd.Series(np.ones(X.shape[0]), name='_weight_')
        else:
            weight = pd.Series(sample_weight, name='_weight_')

        weighted_target = pd.Series(y*weight, name='_weighted_target_')

        df_train = pd.concat([X, y, weight, weighted_target], axis=1)

        return df_train
