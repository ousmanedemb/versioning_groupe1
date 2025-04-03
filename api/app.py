from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib
import os

app = Flask(__name__)
CORS(app)

# Répertoires pour stocker les modèles et les fichiers annexes
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Dictionnaire pour stocker les modèles chargés en mémoire
models = {}
scaler = None

def encode_categorical(df, categorical_columns):
    """
    Encode manuellement les colonnes catégoriques en colonnes binaires.
    Retourne le DataFrame encodé et un dictionnaire de mappage des catégories.
    """
    encoding_mapping = {}
    for col in categorical_columns:
        # Récupérer les valeurs uniques de la colonne
        unique_values = df[col].unique()
        # Créer un mappage des valeurs uniques
        encoding_mapping[col] = {value: f"{col}_{value}" for value in unique_values}
        # Encoder la colonne en colonnes binaires
        for value in unique_values:
            df[encoding_mapping[col][value]] = (df[col] == value).astype(int)
        # Supprimer la colonne catégorique originale
        df.drop(col, axis=1, inplace=True)
    return df, encoding_mapping

def preprocess_data(df, model_type):
    global scaler
    
    # Normalisation des noms de colonnes
    df.columns = df.columns.str.strip().str.lower()
    
    # Vérification des colonnes requises
    required_columns = {'id', 'marque', 'modele', 'annee', 'kilometrage', 'carburant', 'prix'}
    if not required_columns.issubset(df.columns):
        raise ValueError("Le fichier CSV ne contient pas toutes les colonnes requises.")
    
    # Encodage manuel des variables catégoriques
    categorical_columns = ['marque', 'modele', 'carburant']
    df, encoding_mapping = encode_categorical(df, categorical_columns)
    
    # Séparation des features et de la target
    X = df.drop(['prix', 'id'], axis=1)
    y = df['prix']
    
    # Sauvegarde des noms de colonnes avant la transformation
    columns = X.columns
    
    # Transformation logarithmique du prix pour SVM et Linear Regression
    if model_type in ['svm', 'linear-regression']:
        y = np.log(y + 1)  # Évite les valeurs négatives

    # Standardisation des features (sauf pour Random Forest)
    if model_type in ['linear-regression', 'svm']:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, columns, encoding_mapping  # Retourne les données transformées, les colonnes et le mappage

def train_model(model_type, X, y, n_estimators=100, max_depth=10, C=1.0, kernel='rbf'):
    if model_type == 'linear-regression':
        model = LinearRegression()
    elif model_type == 'random-forest':
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif model_type == 'svm':
        model = SVR(C=C, kernel=kernel)
    else:
        raise ValueError("Type de modèle non supporté")
    
    model.fit(X, y)
    return model

@app.route('/api/train', methods=['POST'])
def train():
    try:
        if 'file' not in request.files or 'model' not in request.form:
            return jsonify({'status': 'error', 'message': 'Données manquantes : fichier CSV et type de modèle requis.'}), 400
        
        file = request.files['file']
        model_type = request.form['model']
        
        # Récupération des paramètres optionnels
        n_estimators = int(request.form.get('n_estimators', 100))
        max_depth = int(request.form.get('max_depth', 10))
        C = float(request.form.get('C', 1.0))
        kernel = request.form.get('kernel', 'rbf')
        
        df = pd.read_csv(file)
        if len(df) < 5:
            return jsonify({'status': 'error', 'message': "Dataset trop petit pour l'entraînement."}), 400

        X_scaled, y, columns, encoding_mapping = preprocess_data(df, model_type)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if len(y_test) < 2:
            return jsonify({'status': 'error', 'message': "Taille du jeu de test insuffisante pour le score R²."}), 400
        
        model = train_model(model_type, X_train, y_train, n_estimators, max_depth, C, kernel)
        models[model_type] = model
        
        # Sauvegarde du modèle
        model_filename = os.path.join(MODEL_DIR, f"{model_type}.pkl")
        joblib.dump(model, model_filename)

        # Sauvegarde du scaler si nécessaire
        if model_type in ['linear-regression', 'svm']:
            joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

        # Sauvegarde des colonnes et du mappage d'encodage
        joblib.dump(columns, os.path.join(MODEL_DIR, "columns.pkl"))
        joblib.dump(encoding_mapping, os.path.join(MODEL_DIR, "encoding_mapping.pkl"))

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        return jsonify({'status': 'success', 'metrics': {'train_score': train_score, 'test_score': test_score}})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data['model']
        
        # Vérifier si le modèle existe
        model_filename = os.path.join(MODEL_DIR, f"{model_type}.pkl")
        if not os.path.exists(model_filename):
            return jsonify({'status': 'error', 'message': 'Modèle non entraîné'}), 400
        
        # Charger le modèle, le scaler, les colonnes et le mappage d'encodage
        model = joblib.load(model_filename)
        if model_type in ['linear-regression', 'svm']:
            scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
        encoding_mapping = joblib.load(os.path.join(MODEL_DIR, "encoding_mapping.pkl"))
        
        # Création du dataframe avec les features reçues
        input_data = pd.DataFrame([{key: data[key] for key in ['marque', 'modele', 'annee', 'kilometrage', 'carburant']}])
        
        # Normalisation de la casse
        for column in ['marque', 'modele', 'carburant']:
            if column in input_data.columns:
                input_data[column] = input_data[column].str.capitalize()

        # Encodage manuel des variables catégoriques
        categorical_columns = ['marque', 'modele', 'carburant']
        for col in categorical_columns:
            if col in input_data.columns:
                for value, encoded_col in encoding_mapping[col].items():
                    input_data[encoded_col] = (input_data[col] == value).astype(int)
                input_data.drop(col, axis=1, inplace=True)
        
        # Ajouter les colonnes manquantes avec des valeurs par défaut (0)
        for col in columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Réorganiser les colonnes pour correspondre à l'ordre d'entraînement
        input_data = input_data[columns]
        
        # Standardisation des features
        if model_type in ['linear-regression', 'svm']:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data.values

        # Prédiction
        prediction = model.predict(input_scaled)[0]
        
        # Si le modèle utilise une transformation logarithmique, appliquer l'exponentielle
        if model_type in ['svm', 'linear-regression']:
            prediction = np.exp(prediction) - 1
        
        return jsonify({'status': 'success', 'prediction': float(prediction)})
    
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)