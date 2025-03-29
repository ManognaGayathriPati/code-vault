 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
df = pd.read_csv('house_prices.csv')
print("Missing values in each column:")
print(df.isnull().sum())
imputer = SimpleImputer(strategy='mean')
df[['LotArea', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'TotalBsmtSF', 'FullBath']] = imputer.fit_transform(
    df[['LotArea', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'TotalBsmtSF', 'FullBath']]
)
X = df[['LotArea', 'BedroomAbvGr', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'TotalBsmtSF', 'FullBath']]
y = np.log1p(df['SalePrice'])  # Log transformation for target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Prices (log scale)')
plt.ylabel('Predicted Prices (log scale)')
plt.title('Actual vs Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # Diagonal line
plt.show()
