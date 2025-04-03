FROM node:16-alpine

WORKDIR /app

# Copier uniquement les fichiers package*.json depuis le contexte de construction
COPY package*.json ./
RUN npm install

# Copier tout le reste des fichiers
COPY . .

EXPOSE 3000

CMD ["npm", "start"]
