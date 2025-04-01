import React from 'react';
import type { MLModel } from '../types';

interface ModelSelectionProps {
  selectedModel: MLModel;
  onModelSelect: (model: MLModel) => void;
}

export function ModelSelection({ selectedModel, onModelSelect }: ModelSelectionProps) {
  const models: { id: MLModel; name: string }[] = [
    { id: 'linear-regression', name: 'Régression Linéaire' },
    { id: 'random-forest', name: 'Random Forest' },
    { id: 'svm', name: 'SVM' },
  ];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Sélection du Modèle</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map((model) => (
          <button
            key={model.id}
            className={`p-4 rounded-lg border-2 transition-colors ${
              selectedModel === model.id
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-blue-300'
            }`}
            onClick={() => onModelSelect(model.id)}
          >
            <p className="font-medium">{model.name}</p>
          </button>
        ))}
      </div>
    </div>
  );
}