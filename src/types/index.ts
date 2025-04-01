export interface Car {
  id: number;
  marque: string;
  modele: string;
  annee: number;
  kilometrage: number;
  carburant: string;
  puissance: number;
  prix: number;
}

export interface ModelResult {
  modelName: string;
  mse: number;
  rmse: number;
  r2: number;
  trainingTime: number;
}

export type MLModel = 'linear-regression' | 'random-forest' | 'svm';