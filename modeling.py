#modeling.py
import h2o
from h2o.automl import H2OAutoML
import pandas as pd

def run_h2o_automl(df, target, max_models=20, max_runtime_secs=300):
    """
    Runs H2O AutoML on the provided DataFrame.
    
    :param df: Pandas DataFrame (processed data)
    :param target: String name of the target variable
    :param max_models: Maximum number of models to build
    :param max_runtime_secs: Maximum runtime in seconds
    :return: H2OAutoML object after training
    """
    # Initialize H2O if not already running
    if not h2o.connection():
        h2o.init()

    # Convert the Pandas DataFrame to an H2O Frame
    hf = h2o.H2OFrame(df)
    
    # Identify predictors: All columns except the target
    predictors = [col for col in hf.columns if col != target]
    
    # If the target is categorical in the original Pandas DF, convert it to a factor in H2O
    if df[target].dtype == 'object':
        hf[target] = hf[target].asfactor()

    # Run H2O AutoML
    aml = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=1)
    aml.train(x=predictors, y=target, training_frame=hf)
    
    # Print the leaderboard to show top models
    lb = aml.leaderboard
    print("🏆 H2O AutoML Leaderboard:")
    print(lb)
    
    return aml

if __name__ == '__main__':
    # Example usage:
    # Suppose you have a processed DataFrame 'df_processed' and you want to predict the column 'sales'.
    # You would call the function like this:
    
    # df_processed = pd.read_csv('your_processed_file.csv')
    # aml_model = run_h2o_automl(df_processed, target='sales', max_models=20, max_runtime_secs=300)
    
    # For demonstration, let's create a dummy DataFrame (replace with your real data):
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'sales': [100, 200, 300, 400, 500]
    }
    df_processed = pd.DataFrame(data)
    
    # Run AutoML:
    aml_model = run_h2o_automl(df_processed, target='sales', max_models=5, max_runtime_secs=60)
    
    # Now, extract the best model:
    best_model = aml_model.leader

    # For testing predictions, let's create a dummy test DataFrame:
    df_test = pd.DataFrame({
        'feature1': [6, 7],
        'feature2': [60, 70]
    })

    # Convert the test DataFrame to an H2OFrame and generate predictions:
    test_h2o = h2o.H2OFrame(df_test)
    predictions = best_model.predict(test_h2o)
    predictions_df = predictions.as_data_frame()
    print("Predictions:")
    print(predictions_df.head())
