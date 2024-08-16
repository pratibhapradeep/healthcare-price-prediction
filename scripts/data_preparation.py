import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('../data/healthcare_prices.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Feature engineering
data['feature_interaction'] = data['feature1'] * data['feature2']

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(data[['feature1', 'feature2', 'feature3', 'feature_interaction']])
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save prepared data
pd.DataFrame(X_train).to_csv('../data/X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('../data/X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('../data/y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('../data/y_test.csv', index=False)
