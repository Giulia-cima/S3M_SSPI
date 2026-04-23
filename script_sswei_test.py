import numpy as np
import pandas as pd
from scipy import stats

def calculate_sswei_logic(swe_series):
    """The math core: Fits Gamma and transforms to Normal."""
    n = len(swe_series)
    # Handle zero-snow cases
    zeros = (swe_series <= 0)
    p_zero = np.sum(zeros) / n
    
    if p_zero == 1: # All values are zero
        return pd.Series(0.0, index=swe_series.index)

    non_zero_swe = swe_series[~zeros]
    
    # Fit Gamma distribution
    # Note: If data is very sparse, this may throw a warning
    shape, loc, scale = stats.gamma.fit(non_zero_swe, floc=0)
    cdf_non_zero = stats.gamma.cdf(non_zero_swe, shape, loc, scale)
    
    # Probability adjustment
    h_x = np.zeros(n)
    h_x[zeros] = p_zero
    h_x[~zeros] = p_zero + (1 - p_zero) * cdf_non_zero
    
    # Transform to Z-score, clipping to avoid infinity at 0 or 1
    h_x = np.clip(h_x, 1e-6, 0.999999)
    return pd.Series(stats.norm.ppf(h_x), index=swe_series.index)

# --- 1. Prepare Data ---
# Assuming 'df' has a DatetimeIndex and a column 'SWE'
# Example: df = pd.read_csv('your_data.csv', index_col=0, parse_dates=True)

# Define the winter months
winter_months = [10, 11, 12, 1, 2, 3]

# Filter for Oct-March
winter_df = df[df.index.month.isin(winter_months)].copy()

# --- 2. Group by Month and Apply ---
# transform(calculate_sswei_logic) ensures the output matches the original length
winter_df['SSWEI'] = (
    winter_df.groupby(winter_df.index.month)['SWE']
    .transform(calculate_sswei_logic)
)

# --- 3. Optional: Re-merge into main dataframe ---
# This puts the results back into your full timeline, with NaNs for April-Sept
df['SSWEI'] = winter_df['SSWEI']

print(df.loc['1996-10-01':'1997-04-01'])
