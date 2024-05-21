import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create the DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'marks': [85, 92, 78, 90, 88],
    'cgpa': [3.8, 4.0, 3.2, 3.9, 3.7],
    'percentage': [85, 92, 78, 90, 88]
}
df = pd.DataFrame(data)
print("Initial DataFrame:")
print(df)

# Step 2: Data Preprocessing
numeric_columns = ['marks', 'cgpa', 'percentage']
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print("\nDataFrame after Standardization:")
print(df)

# Step 3: Model Training
X = df[['cgpa', 'percentage']]
y = df['marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation:")
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')







     
     




  
          
        
    
    
    