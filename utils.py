#function for VIF to check for multi-collinearity
import statsmodels.stats.outliers_influence as sm
import sklearn.feature_selection as sfs
import pandas as pd

def variance_threshold_selection_remove_cols(df, threshold):
  """
  Selects features with a variance greater than a threshold value and returns a list of columns to be removed.

  Args:
    df: The DataFrame to select features from.
    threshold: The minimum variance threshold.

  Returns:
    A list of columns to be removed.
  """

  selector = sfs.VarianceThreshold(threshold=threshold)
  selected_features = selector.fit_transform(df)

  cols_to_remove = [col for col in df.columns if col not in selected_features.columns]

  return cols_to_remove
  