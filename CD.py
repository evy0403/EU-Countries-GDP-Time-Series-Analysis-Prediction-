# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv(r"C:\Users\Evelyn\Projects\DA\Data\ed.csv")

print(df.head())

# Data Preprocessing
print("Missing values:")
print(df.isnull().sum())  

df = df.dropna(subset=['Year', 'Germany', 'France', 'United Kingdom'])


df['Year'] = df['Year'].astype(int)


df['Germany'] = pd.to_numeric(df['Germany'], errors='coerce')
df['France'] = pd.to_numeric(df['France'], errors='coerce')
df['United Kingdom'] = pd.to_numeric(df['United Kingdom'], errors='coerce')


df = df.dropna(subset=['Germany', 'France', 'United Kingdom'])


sns.set(style="whitegrid")

# Plot GDP trends for selected countries
plt.figure(figsize=(12, 6))
plt.plot(df['Year'], df['Germany'], label='Germany', color='blue')
plt.plot(df['Year'], df['France'], label='France', color='red')
plt.plot(df['Year'], df['United Kingdom'], label='United Kingdom', color='green')

plt.xlabel('Year')
plt.ylabel('GDP in USD')
plt.title('GDP Time Series for Selected EU Countries')


plt.legend()

# Show the plot
plt.show()

# Preparing Data for Prediction
# We will use GDP of Germany (as an example) for prediction based on years
X = df[['Year']]  # Feature: Year
y = df['Germany']  # Target: GDP of Germany

# Train-Test Split
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Train-test split shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Linear Regression Model
# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the GDP values for the test set
y_pred = model.predict(X_test)

# Print the coefficients and intercept
print("Coefficient: ", model.coef_)
print("Intercept: ", model.intercept_)

# Model Evaluation
# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualizing Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual GDP')
plt.plot(X_test, y_pred, color='red', label='Predicted GDP')
plt.xlabel('Year')
plt.ylabel('GDP in USD')
plt.title('Actual vs Predicted GDP for Germany')
plt.legend()
plt.show()

# Future Prediction (Optional)
# Predict GDP for future years (e.g., 2025-2030)
future_years = pd.DataFrame({'Year': np.arange(2025, 2031)})
future_gdp = model.predict(future_years)

# Display predicted GDP values for future years
future_years['Predicted GDP'] = future_gdp
print(future_years)
