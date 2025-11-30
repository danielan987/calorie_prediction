# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error

# Import train.csv.gz data with your folder path  
data = pd.read_csv('', compression='gzip')

# Transform and split data 
data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})
X = data.drop(['id', 'Calories'], axis=1)
y=data['Calories']
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Run a random forest model
rf_model = RandomForestRegressor(n_estimators=250, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Evaluate regression metrics 
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

msle = mean_squared_log_error(y_test, y_pred)
rmsle = np.sqrt(msle)
print("RMS:E:", rmsle)

# Import test.csv.gz data with your folder path 
test_data = pd.read_csv('', compression='gzip')

# Transform data 
test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0})
X_test = test_data.drop(['id'], axis=1)

# Predict calories on test data 
test_predictions = rf_model.predict(X_test)

# Combine predictions with test IDs
output = pd.DataFrame({
    'id': test_data['id'],
    'Calories': test_predictions
})

# Export prediction output with your folder path
output.to_csv('', index=False, compression='gzip')
