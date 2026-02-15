import pickle, os, json, random, sys, argparse, joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    timestamp = args.timestamp
    model_version = f'model_{timestamp}_lr_model'
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'model_{timestamp}_lr_model.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Create evaluation data
    X, y = make_classification(
        n_samples=800,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=123
    )

    y_pred = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "F1_Score": f1_score(y, y_pred)
    }
    
    os.makedirs('metrics', exist_ok=True)
    metrics_path = f'metrics/{timestamp}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✅ Model evaluation complete. Metrics saved to: {metrics_path}")
    print(metrics)
