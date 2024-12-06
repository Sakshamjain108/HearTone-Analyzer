import pandas as pd
import matplotlib.pyplot as plt
# Load your dataset
data = pd.read_csv('HoerstatAalen_HTLdata.csv')

# Select columns with HTL frequencies and the target variable 'HearAids'
htl_columns = [col for col in data.columns if 'HTL' in col]

# Calculate correlations with the target
correlations = data[htl_columns + ['HearAids']].corr()['HearAids'].drop('HearAids')

# Calculate weights
abs_correlations = correlations.abs()
weights = abs_correlations / abs_correlations.sum()

# Display the weights
print(weights)


# Calculate the average weights for each frequency across L and R
frequencies = ['250', '500', '1000', '2000', '3000', '4000', '6000', '8000']
average_weights = {}

for freq in frequencies:
    avg_weight = (weights[f'R_{freq}_HTL'] + weights[f'L_{freq}_HTL']) / 2
    average_weights[f'{freq} Hz'] = avg_weight
dfx = pd.DataFrame.from_dict(average_weights, orient='index')
print(dfx)

# Convert to a DataFrame for easy plotting
from scipy.interpolate import make_interp_spline
import numpy as np

# Convert the frequencies and average weights to arrays for interpolation
frequencies_numeric = np.array([250, 500, 1000, 2000, 3000, 4000, 6000, 8000])
average_weights_array = np.array(list(average_weights.values()))

# Create a spline interpolation of the data
spl = make_interp_spline(frequencies_numeric, average_weights_array, k=3)
frequencies_smooth = np.linspace(frequencies_numeric.min(), frequencies_numeric.max(), 500)
weights_smooth = spl(frequencies_smooth)

# Plot the smooth line
plt.figure(figsize=(10, 6))
plt.plot(frequencies_smooth, weights_smooth, color='skyblue', linewidth=2)
plt.title('Smooth Average Weight of Each Frequency on HearAids')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Average Weight')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()