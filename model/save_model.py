import pickle
from model import train_model

# Train and save the model
model = train_model()
with open('life_expectancy_model.pkl', 'wb') as file:
    pickle.dump(model, file)