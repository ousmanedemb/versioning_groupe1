from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables
models = {}
scaler = None


def encode_categorical(df, categorical_columns):
    """One-hot encode categorical columns manually"""
    encoding_mapping = {}
    for col in categorical_columns:
        unique_values = df[col].unique()
        encoding_mapping[col] = {value: f"{col}_{value}" for value in unique_values}
        for value in unique_values:
            df[encoding_mapping[col][value]] = (df[col] == value).astype(int)
        df.drop(col, axis=1, inplace=True)
    return df, encoding_mapping


def preprocess_data(df, model_type):
    """Preprocess data including encoding and scaling"""
    global scaler

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Validate required columns
    required_columns = {'id', 'marque', 'modele', 'annee', 'kilometrage', 'carburant', 'prix'}
    if not required_columns.issubset(df.columns):
        raise ValueError("Missing required columns in CSV file.")

    # Encode categorical variables
    categorical_columns = ['marque', 'modele', 'carburant']
    df, encoding_mapping = encode_categorical(df, categorical_columns)

    # Separate features and target
    X = df.drop(['prix', 'id'], axis=1)
    y = df['prix']
    columns = X.columns

    # Log transform for certain models
    if model_type in ['svm', 'linear-regression']:
        y = np.log(y + 1)

    # Standardize features
    if model_type in ['linear-regression', 'svm']:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, columns, encoding_mapping


def train_model(model_type, X, y, params):
    """Train and return the specified model"""
    if model_type == 'linear-regression':
        model = LinearRegression()
    elif model_type == 'random-forest':
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            random_state=42
        )
    elif model_type == 'svm':
        model = SVR(
            C=params.get('C', 1.0),
            kernel=params.get('kernel', 'rbf')
        )
    else:
        raise ValueError("Unsupported model type")

    model.fit(X, y)
    return model


def generate_evaluation_plots(model, X_test, y_test, model_type, columns):
    """Generate evaluation plots and return as base64 encoded images"""
    plots = {}

    # Make predictions
    y_pred = model.predict(X_test)

    # Reverse log transform if needed
    if model_type in ['svm', 'linear-regression']:
        y_test = np.exp(y_test) - 1
        y_pred = np.exp(y_pred) - 1

    # Plot 1: Actual vs Predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plots['actual_vs_predicted'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Plot 2: Residual plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - y_pred
    sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plots['residual_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # Plot 3: Feature importance (for Random Forest)
    if model_type == 'random-forest' and hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15 features
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [columns[i] for i in indices])
        plt.xlabel('Relative Importance')

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plots['feature_importance'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

    return plots


@app.route('/api/train', methods=['POST'])
def train():
    try:
        # Validate request
        if 'file' not in request.files or 'model' not in request.form:
            return jsonify({'status': 'error', 'message': 'CSV file and model type are required.'}), 400

        # Get parameters
        file = request.files['file']
        model_type = request.form['model']
        params = {
            'n_estimators': int(request.form.get('n_estimators', 100)),
            'max_depth': int(request.form.get('max_depth', 10)),
            'C': float(request.form.get('C', 1.0)),
            'kernel': request.form.get('kernel', 'rbf')
        }

        # Load and validate data
        df = pd.read_csv(file)
        if len(df) < 10:
            return jsonify({'status': 'error', 'message': 'Dataset too small for training.'}), 400

        # Preprocess data
        X, y, columns, encoding_mapping = preprocess_data(df, model_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if len(y_test) < 2:
            return jsonify({'status': 'error', 'message': 'Insufficient test data for evaluation.'}), 400

        # Train model
        model = train_model(model_type, X_train, y_train, params)
        models[model_type] = model

        # Save artifacts
        joblib.dump(model, os.path.join(MODEL_DIR, f"{model_type}.pkl"))
        if model_type in ['linear-regression', 'svm']:
            joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(columns, os.path.join(MODEL_DIR, "columns.pkl"))
        joblib.dump(encoding_mapping, os.path.join(MODEL_DIR, "encoding_mapping.pkl"))

        # Generate evaluation metrics
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        # Generate evaluation plots
        plots = generate_evaluation_plots(model, X_test, y_test, model_type, columns)

        return jsonify({
            'status': 'success',
            'metrics': {
                'train_score': train_score,
                'test_score': test_score,
                'r2_score': r2_score(y_test, model.predict(X_test))
            },
            'plots': plots
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data['model']

        # Validate model exists
        model_path = os.path.join(MODEL_DIR, f"{model_type}.pkl")
        if not os.path.exists(model_path):
            return jsonify({'status': 'error', 'message': 'Model not trained yet.'}), 400

        # Load artifacts
        model = joblib.load(model_path)
        if model_type in ['linear-regression', 'svm']:
            scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
        encoding_mapping = joblib.load(os.path.join(MODEL_DIR, "encoding_mapping.pkl"))

        # Prepare input data
        input_data = pd.DataFrame([{
            'marque': data.get('marque', ''),
            'modele': data.get('modele', ''),
            'annee': data.get('annee', 0),
            'kilometrage': data.get('kilometrage', 0),
            'carburant': data.get('carburant', '')
        }])

        # Capitalize categorical values
        for col in ['marque', 'modele', 'carburant']:
            if col in input_data.columns:
                input_data[col] = input_data[col].str.capitalize()

        # One-hot encode categorical variables
        for col in ['marque', 'modele', 'carburant']:
            if col in input_data.columns:
                for value, encoded_col in encoding_mapping[col].items():
                    input_data[encoded_col] = (input_data[col] == value).astype(int)
                input_data.drop(col, axis=1, inplace=True)

        # Add missing columns with 0 values
        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns
        input_data = input_data[columns]

        # Scale features if needed
        if model_type in ['linear-regression', 'svm']:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data.values

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        if model_type in ['svm', 'linear-regression']:
            prediction = np.exp(prediction) - 1

        return jsonify({
            'status': 'success',
            'prediction': float(prediction),
            'currency': 'EUR'  # You can make this configurable
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    try:
        models = []
        for file in os.listdir(MODEL_DIR):
            if file.endswith('.pkl') and not file.startswith(('scaler', 'columns', 'encoding')):
                models.append(file.replace('.pkl', ''))
        return jsonify({'status': 'success', 'models': models})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)