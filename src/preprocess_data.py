import pandas as pd
# Setting up PyCaret environment
# Target variable is 'Outcome'
from pycaret.classification import *


# Load dataset
data = pd.read_csv('../data/iris.csv')
clf = setup(data=data, target='species', session_id=123)

# Save the setup configuration for reuse
save_experiment('models/preprocessing_config.pkl')


