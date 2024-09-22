import pandas as pd
from pycaret.classification import *
from pycaret.datasets import get_data

# Load the dataset
df = get_data('iris')

# Save DataFrame to CSV
df.to_csv('../data/iris.csv', index=False)
