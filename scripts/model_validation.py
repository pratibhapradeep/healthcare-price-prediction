import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the prepared test data
X_test = pd.read_csv('../data/X_test.csv')
y_test = pd.read_csv('../data/y_test.csv').values.ravel()  # Convert to 1D array

# List of saved models
model_filenames = ['linear_regression_model.pkl', 'random_forest_model.pkl', 'gradient_boosting_model.pkl']
model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']

# Validate each model
for model_name, model_file in zip(model_names, model_filenames):
    # Load the model
    model = joblib.load(f'../models/{model_file}')

    # Make predictions
    y_pred = model.predict(X_test).ravel()  # Ensure y_pred is 1D

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f'{model_name}:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    print('---')

    # Save predictions for further analysis
    predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions.to_csv(f'../data/{model_name.replace(" ", "_").lower()}_predictions.csv', index=False)
