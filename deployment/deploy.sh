#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [direct|docker]"
  exit 1
fi

MODE=$1

case $MODE in
  direct)
    echo "Déploiement direct sur l'OS..."

    echo "Déploiement du backend..."
    cd ../api || exit
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    nohup python app.py > backend.log 2>&1 &
    echo "Backend déployé sur http://localhost:5000"

    echo "Déploiement du frontend..."
    cd ../frontend || exit
    npm install
    nohup npm start > frontend.log 2>&1 &
    echo "Frontend déployé sur http://localhost:3000"
    ;;

  docker)
    echo "Build et déploiement avec Docker Compose..."
    cd /home/meta/code_factory/versioning_groupe1/deployment || exit
    docker-compose up --build -d
    echo "Applications déployées avec Docker Compose."
    echo "Frontend: http://localhost:3000"
    echo "Backend: http://localhost:5000"
    ;;

  *)
    echo "Mode inconnu: $MODE"
    echo "Usage: $0 [direct|docker]"
    exit 1
    ;;
esac
