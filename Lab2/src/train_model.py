import mlflow, datetime, os, pickle, random, sys, argparse
from joblib import dump
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow.sklearn

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    timestamp = args.timestamp
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=random.randint(1000, 2000),
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42,
        shuffle=True,
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure data folder exists
    os.makedirs('data', exist_ok=True)
    pickle.dump(X, open('data/data.pickle', 'wb'))
    pickle.dump(y, open('data/target.pickle', 'wb'))
    
    # Setup MLflow
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Synthetic Classification Data"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"    
    experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id, run_name=dataset_name):
        
        params = {
            "dataset_name": dataset_name,
            "number_of_samples": X.shape[0],
            "number_of_features": X.shape[1]
        }
        mlflow.log_params(params)
        
        # ✅ Logistic Regression model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metrics({'Accuracy': acc, 'F1_Score': f1})
        mlflow.sklearn.log_model(model, "LogisticRegressionModel")
        
        os.makedirs('models', exist_ok=True)
        model_version = f'model_{timestamp}'
        model_filename = f'{model_version}_lr_model.joblib'
        dump(model, os.path.join('models', model_filename))
        
        print(f"✅ Model training completed. Saved as: {model_filename}")
        print(f"📊 Accuracy: {acc:.3f}, F1 Score: {f1:.3f}")

