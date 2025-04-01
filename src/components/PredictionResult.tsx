import React from 'react';
import { TrendingUp } from 'lucide-react';

interface PredictionResultProps {
  predictedPrice: number;
}

export function PredictionResult({ predictedPrice }: PredictionResultProps) {
  const formattedPrice = new Intl.NumberFormat('fr-FR', {
    style: 'currency',
    currency: 'XOF',
    maximumFractionDigits: 0
  }).format(predictedPrice);

  return (
    <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
      <div className="flex items-center space-x-3 mb-4">
        <TrendingUp className="w-6 h-6 text-green-500" />
        <h3 className="text-lg font-semibold">Résultat de la Prédiction</h3>
      </div>
      <p className="text-3xl font-bold text-center text-green-600">{formattedPrice}</p>
      <p className="text-sm text-gray-500 text-center mt-2">Prix estimé du véhicule</p>
    </div>
  );
}