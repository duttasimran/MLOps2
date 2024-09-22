import optuna
import pandas as pd
from pycaret.classification import *

data = pd.read_csv('../data/iris.csv')

def perform_hyperparameter_tuning_and_logging(best_model):
    # tune model optuna
    print("Tuning with Optuna")
    tuned_Optuna=tune_model(best_model, search_library = 'optuna',n_iter=50)
    results=pull()
    accuracy_optuna=results['Accuracy'][0]

    # tune model scikit-optimize
    print("Tuning with scikit-optimize")
    tuned_scikit=tune_model(best_model, search_library = 'scikit-optimize',n_iter=50)
    results=pull()
    accuracy_scikit=results['Accuracy'][0]

    #save efficient model
    if accuracy_optuna>=accuracy_scikit:
        return tuned_Optuna 
    else:
        return tuned_scikit

    

if __name__ == "__main__":
    with open('models/preprocessing_config.pkl', 'rb') as f:
        load_experiment(f,data=data)

    # Compare multiple models and choose the best one based on cross-validation
    best_model = compare_models(include=['rf', 'dt', 'lightgbm', 'et'])
    
    # Print the best model selected
    print(best_model)
    predict_model(best_model)
    save_model(best_model,  'models/best')
    
    #Performing Hyperparameter tuning to improve accuracy
    final_model1=perform_hyperparameter_tuning_and_logging(best_model)
    print(final_model1)
    
    # Finalize the model to lock it in for future predictions
    final_model = finalize_model(final_model1)
    
    # save model
    save_model(final_model,  'models/speciesPrediction')
