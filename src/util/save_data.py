import dill as pickle

def save_object(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)