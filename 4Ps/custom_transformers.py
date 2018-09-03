"""
Class that selects best features based on correlation

Taken from Rafael Wampfler @ETH Zurich
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FindCorrelation(BaseEstimator, TransformerMixin):
    '''
    Remove pairwise correlations beyond threshold.
    '''
    def __init__(self, threshold=0.9):
       self.threshold = threshold

    def fit(self, X, y=None):
        '''
        Produce binary array for filtering columns in feature array.
        Remember to transpose the correlation matrix so is
        column major.
        Loop through columns in (n_features,n_features) correlation matrix.
        Determine rows where value is greater than threshold.
        For the candidate pairs, one must be removed. Determine which feature
        has the larger average correlation with all other features and remove it.
        Remember, matrix is symmetric so shift down by one row per column as
        iterate through.
        '''
        # Calculate the correlation matrix of the predictors
        self.correlated = np.zeros(X.shape[1], dtype=bool)
        self.corr_mat =  np.corrcoef(X.T)
        abs_corr_mat = np.abs(self.corr_mat)

        stop = False
        while(not stop):
            # Determine the two predictors associated with the largest absolute pairwise correlation (call them predictors A and B).
            max_m = 0
            pairs = None

            for i, col in enumerate(abs_corr_mat.T):
                if self.correlated[i] == True:
                    continue

                for j, val in enumerate(col):
                    if (self.correlated[j] == True) or (i == j):
                        continue

                    if (val > self.threshold) and (val > max_m):
                        max_m = val
                        pairs = [i, j]

            if pairs is not None:
                # Determine the average correlation between A and the other variables. Do the same for predictor B.
                A = np.mean(abs_corr_mat.T[:, pairs[0]])
                B = np.mean(abs_corr_mat.T[:, pairs[1]])

                # If A has a larger average correlation, remove it; otherwise, remove predictor B.
                if A > B:
                    self.correlated[pairs[0]] = True

                else:
                    self.correlated[pairs[1]] = True
            else:
                stop = True

        """
        for i, col in enumerate(abs_corr_mat.T):
            corr_rows = np.where(col[i+1:] > self.threshold)[0]
            avg_corr = np.mean(col)

            if len(corr_rows) > 0:
                for j in corr_rows:
                    if np.mean(abs_corr_mat.T[:, j]) > avg_corr:
                        self.correlated[j] = True
                    else:
                        self.correlated[i] = True
        """

        return self

    def transform(self, X, y=None):
        '''
        Mask the array with the features flagged for removal
        '''
        return X.T[~self.correlated].T

    def get_feature_names(self, input_features=None):
        return input_features[~self.correlated]

    def get_feature_index(self):
        return ~self.correlated
