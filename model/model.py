import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model():
    # Load data
    life = pd.read_csv('C:/Users/varka/OneDrive/Desktop/life expectancy/Life Expectancy Data.csv')
    
    # Preprocess Status (Developing: 0, Developed: 1)
    life['Status'] = life['Status'].map({'Developing': 0, 'Developed': 1})
    
    # Features and target
    x = life.drop(columns=['Life expectancy ', 'Country'], axis=1)
    y = life['Life expectancy ']
    
    # Handle missing values (simple imputation with mean)
    x = x.fillna(x.mean())
    y = y.fillna(y.mean())
    
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    
    # Train model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    return model