import React, { useState } from 'react';
import type { MLModel } from '../types';

interface ModelParametersProps {
  model: MLModel;
  onStartTraining: (params: Record<string, string>) => void;
}

export function ModelParameters({ model, onStartTraining }: ModelParametersProps) {
  // Initialiser les valeurs par défaut pour chaque modèle
  const defaultParams: Record<string, string> = {
    alpha: "0.01",
    n_estimators: "100",
    max_depth: "10",
    C: "1.0",
    kernel: "rbf",
  };

  // État local pour stocker les paramètres
  const [params, setParams] = useState<Record<string, string>>(defaultParams);

  // Fonction pour mettre à jour les paramètres dynamiquement
  const handleParamChange = (key: string, value: string) => {
    setParams(prevParams => ({ ...prevParams, [key]: value }));
  };

  // Envoi des paramètres lors du clic sur "Lancer l'entraînement"
  const handleTrainClick = () => {
    onStartTraining(params);
  };

  // Générer dynamiquement les champs des paramètres en fonction du modèle sélectionné
  const getModelParams = () => {
    switch (model) {
      case 'linear-regression':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Régularisation (alpha)
              </label>
              <input
                type="number"
                min="0"
                step="0.01"
                value={params.alpha}
                onChange={(e) => handleParamChange('alpha', e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
          </div>
        );
      case 'random-forest':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Nombre d'arbres
              </label>
              <input
                type="number"
                min="1"
                value={params.n_estimators}
                onChange={(e) => handleParamChange('n_estimators', e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Profondeur maximale
              </label>
              <input
                type="number"
                min="1"
                value={params.max_depth}
                onChange={(e) => handleParamChange('max_depth', e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
          </div>
        );
      case 'svm':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Paramètre C
              </label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={params.C}
                onChange={(e) => handleParamChange('C', e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Kernel
              </label>
              <select
                value={params.kernel}
                onChange={(e) => handleParamChange('kernel', e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="linear">Linéaire</option>
                <option value="rbf">RBF</option>
                <option value="poly">Polynomial</option>
              </select>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
      <h3 className="text-lg font-semibold mb-4">Paramètres du Modèle</h3>
      {getModelParams()}
      <button
        onClick={handleTrainClick}
        className="mt-6 w-full bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors"
      >
        Lancer l'Entraînement
      </button>
    </div>
  );
}