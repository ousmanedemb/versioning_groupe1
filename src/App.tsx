import React, { useState } from 'react';
import { FileUpload } from './components/FileUpload';
import { ModelSelection } from './components/ModelSelection';
import { ModelParameters } from './components/ModelParameters';
import { TrainingProgress } from './components/TrainingProgress';
import { ModelTesting } from './components/ModelTesting';
import { PredictionResult } from './components/PredictionResult';
import type { MLModel, Car } from './types';
import { Car as CarIcon } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<MLModel>('linear-regression');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isModelTrained, setIsModelTrained] = useState(false);
  const [predictedPrice, setPredictedPrice] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<{ train_score: number; test_score: number } | null>(null);
  const [modelParams, setModelParams] = useState<Record<string, string>>({});

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setIsModelTrained(false);
    setPredictedPrice(null);
    setError(null);
    setTrainingMetrics(null);
  };

  const handleStartTraining = async (params: Record<string, string>) => {
    if (!selectedFile) return;
  
    setIsTraining(true);
    setError(null);
    setPredictedPrice(null);
    setTrainingMetrics(null);
  
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('model', selectedModel);
  
      // Ajouter les paramètres du modèle dans le FormData
      Object.entries(params).forEach(([key, value]) => {
        formData.append(key, value);
      });
  
      const response = await axios.post(`${API_URL}/train`, formData);
  
      if (response.data.status === 'success') {
        setIsModelTrained(true);
        setTrainingMetrics(response.data.metrics);
      } else {
        throw new Error(response.data.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Une erreur est survenue');
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredict = async (data: Omit<Car, 'id' | 'prix'>) => {
    try {
      setError(null);
      const response = await axios.post(`${API_URL}/predict`, {
        ...data,
        model: selectedModel
      });

      if (response.data.status === 'success') {
        setPredictedPrice(response.data.prediction);
      } else {
        throw new Error(response.data.message);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Une erreur est survenue');
      setPredictedPrice(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center space-x-3">
            <CarIcon className="w-8 h-8 text-blue-500" />
            <h1 className="text-2xl font-bold text-gray-900">
              Prédiction Prix Voitures
            </h1>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow p-6 space-y-8">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative">
              {error}
            </div>
          )}

          <div>
            <h2 className="text-xl font-semibold mb-4">
              Chargement des Données
            </h2>
            <FileUpload onFileSelect={handleFileSelect} />
            {selectedFile && (
              <p className="mt-2 text-sm text-gray-600">
                Fichier sélectionné: {selectedFile.name}
              </p>
            )}
          </div>

          {selectedFile && (
            <>
              <ModelSelection
                selectedModel={selectedModel}
                onModelSelect={setSelectedModel}
              />
              <ModelParameters
                model={selectedModel}
                onStartTraining={handleStartTraining}
              />
              {isTraining && <TrainingProgress progress={trainingProgress} />}

              {trainingMetrics && (
                <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded">
                  <h3 className="text-lg font-semibold">Résultats de l'entraînement</h3>
                  <p>Précision sur l'entraînement : <strong>{(trainingMetrics.train_score * 100).toFixed(2)}%</strong></p>
                  <p>Précision sur le test : <strong>{(trainingMetrics.test_score * 100).toFixed(2)}%</strong></p>
                </div>
              )}

              {isModelTrained && (
                <>
                  <ModelTesting onPredict={handlePredict} />
                  {predictedPrice !== null && (
                    <PredictionResult predictedPrice={predictedPrice} />
                  )}
                </>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;