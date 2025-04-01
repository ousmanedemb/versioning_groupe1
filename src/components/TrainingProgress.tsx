import React from 'react';
import { Loader2 } from 'lucide-react';

interface TrainingProgressProps {
  progress: number;
}

export function TrainingProgress({ progress }: TrainingProgressProps) {
  return (
    <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
      <div className="flex items-center space-x-3 mb-4">
        <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
        <h3 className="text-lg font-semibold">Entraînement en cours...</h3>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2.5">
        <div
          className="bg-blue-500 h-2.5 rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        ></div>
      </div>
      <p className="text-sm text-gray-600 mt-2 text-center">{progress}% complété</p>
    </div>
  );
}