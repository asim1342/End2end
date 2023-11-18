
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# pickle
import pickle



def pkl_it(model):
    pickle.dump(model, open('regmodel.pkl', 'wb'))

def load_pkl(path):
    pickled_model = pickle.load(open(path, 'rb'))
    return pickled_model

def run_linear_regression():
    # Generating dummy data for regression
    X, y = make_regression(n_samples=100, n_features=1, noise=10)

    # Creating a linear regression model
    model = LinearRegression()

    # Training the model
    model.fit(X, y)

    # Generating some new data points for prediction
    X_new = np.linspace(-3, 3, 100).reshape(-1, 1)

    # Making predictions
    results = model.predict(X_new)

    # pickle it
    pkl_it(model)

    #unpickle it
    model = load_pkl(path='regmodel.pkl')
    return results, model