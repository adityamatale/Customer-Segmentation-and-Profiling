import pickle

# Load the model
with open("final_model.sav", 'rb') as f:
    model = pickle.load(f)

# Check if the model has the 'monotonic_cst' attribute and delete it if present
if hasattr(model, 'monotonic_cst'):
    del model.monotonic_cst

# Use the model for predictions
