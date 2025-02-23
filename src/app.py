import sys
from flask import Flask
from flask_cors import CORS
from routes.detectionRoutes import detection_api
import os

# Création du dossier 'uploads' s'il n'existe pas
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
CORS(app)

# Configuration du dossier d'upload
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Enregistrer les routes
app.register_blueprint(detection_api, url_prefix="/api/detection")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
