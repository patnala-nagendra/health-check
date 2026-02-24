from numpy import loadtxt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# 1. Load dataset function
def load_dataset(file_path):
    dataset = loadtxt(file_path, delimiter=',')
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    return X, y



# 2. Split dataset function
def split_data(X, y, test_size=0.33, seed=7):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed
    )
    return X_train, X_test, y_train, y_test



# 3. Train model function
def train_model(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model



# 4. Prediction function
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions



# 5. Main execution function
def main():
    file_path = "pima-indians-diabetes.csv"
    
    # Load data
    X, y = load_dataset(file_path)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Predict
    predictions = make_predictions(model, X_test)
    
    print("Test Data:")
    print(X_test)
    print("\nPredictions:")
    print(predictions)


# Run program
if __name__ == "__main__":
    main()
 
