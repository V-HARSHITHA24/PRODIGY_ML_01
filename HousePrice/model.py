import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("data.csv")

data = data.drop(['sqft_above'], axis=1)


data = data[data['price'] < data['price'].quantile(0.94)]


#features = data[['sqft_living', 'bedrooms', 'bathrooms']]
features = data[['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'yr_built']]
target = data['price']


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


#X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(scaled_features, data['price'], test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

model = LinearRegression()

model.fit(X_train, y_train)
print("Model trained successfully!")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Drop non-numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

# Plot the heatmap
import seaborn as sns
import matplotlib.pyplot as plt

"""plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()"""

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(scaled_features)

#print(f"Original features shape: {scaled_features.shape}")
#print(f"Polynomial features shape: {X_poly.shape}")

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(scaled_features)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, target, test_size=0.2, random_state=42)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_poly)
print("Polynomial model trained successfully!")

y_pred_poly = model_poly.predict(X_test_poly)

mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
r2_poly = r2_score(y_test_poly, y_pred_poly)

# Results
print(f"Polynomial Model - Mean Squared Error: {mse_poly:.2f}")
print(f"Polynomial Model - R² Score: {r2_poly:.2f}")

from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

ridge_r2 = r2_score(y_test, ridge_pred)
print(f"Ridge R² Score: {ridge_r2:.2f}")

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

lasso_r2 = r2_score(y_test, lasso_pred)
print(f"Lasso R² Score: {lasso_r2:.2f}")














