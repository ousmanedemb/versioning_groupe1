import React, { useState, useMemo } from 'react';
import { Car as CarIcon } from 'lucide-react';
import type { Car } from '../types';

interface ModelTestingProps {
  onPredict: (data: Omit<Car, 'id' | 'prix'>) => void;
}

// Base de données des marques et modèles
const carDatabase = {
  'Renault': ['Clio', 'Megane', 'Captur', 'Kadjar', 'Scenic', 'Talisman', 'Twingo'],
  'Peugeot': ['208', '308', '2008', '3008', '5008', '508'],
  'Citroën': ['C3', 'C4', 'C5', 'Berlingo', 'C3 Aircross', 'C5 Aircross'],
  'Volkswagen': ['Golf', 'Polo', 'Tiguan', 'Passat', 'T-Roc', 'T-Cross'],
  'BMW': ['Série 1', 'Série 2', 'Série 3', 'Série 4', 'Série 5', 'X1', 'X3', 'X5'],
  'Mercedes': ['Classe A', 'Classe B', 'Classe C', 'Classe E', 'GLA', 'GLB', 'GLC'],
  'Audi': ['A1', 'A3', 'A4', 'A5', 'A6', 'Q3', 'Q5', 'Q7'],
  'Toyota': ['Yaris', 'Corolla', 'RAV4', 'C-HR', 'Prius', 'Camry'],
  'Ford': ['Fiesta', 'Focus', 'Puma', 'Kuga', 'Mustang', 'Explorer'],
  'Opel': ['Corsa', 'Astra', 'Crossland', 'Grandland', 'Mokka']
};

export function ModelTesting({ onPredict }: ModelTestingProps) {
  const [formData, setFormData] = useState<Omit<Car, 'id' | 'prix'>>({
    marque: Object.keys(carDatabase)[0],
    modele: carDatabase[Object.keys(carDatabase)[0]][0],
    annee: new Date().getFullYear(),
    kilometrage: 0,
    carburant: 'essence',
    puissance: 0
  });

  const availableModels = useMemo(() => {
    return carDatabase[formData.marque] || [];
  }, [formData.marque]);

  const handleMarqueChange = (marque: string) => {
    setFormData({
      ...formData,
      marque,
      modele: carDatabase[marque][0] // Sélectionner le premier modèle par défaut
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict(formData);
  };

  return (
    <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
      <div className="flex items-center space-x-3 mb-6">
        <CarIcon className="w-6 h-6 text-blue-500" />
        <h3 className="text-lg font-semibold">Tester le Modèle</h3>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Marque</label>
            <select
              value={formData.marque}
              onChange={(e) => handleMarqueChange(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              {Object.keys(carDatabase).map((marque) => (
                <option key={marque} value={marque}>
                  {marque}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Modèle</label>
            <select
              value={formData.modele}
              onChange={(e) => setFormData({ ...formData, modele: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              {availableModels.map((modele) => (
                <option key={modele} value={modele}>
                  {modele}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Année</label>
            <input
              type="number"
              required
              min="1900"
              max={new Date().getFullYear()}
              value={formData.annee}
              onChange={(e) => setFormData({ ...formData, annee: parseInt(e.target.value) })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Kilométrage</label>
            <input
              type="number"
              required
              min="0"
              value={formData.kilometrage}
              onChange={(e) => setFormData({ ...formData, kilometrage: parseInt(e.target.value) })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Carburant</label>
            <select
              value={formData.carburant}
              onChange={(e) => setFormData({ ...formData, carburant: e.target.value })}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="essence">Essence</option>
              <option value="diesel">Diesel</option>
              <option value="hybride">Hybride</option>
              <option value="electrique">Electrique</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          className="w-full bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors"
        >
          Prédire le Prix
        </button>
      </form>
    </div>
  );
}