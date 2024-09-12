from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from loguru import logger
import uvicorn
import numpy as np
import pandas as pd

# Initialisation de FastAPI
app = FastAPI()

# Définition d'un modèle pour les données d'entrée
class PredictionData(BaseModel):
    surface: float


# Définition d'un modèle pour les données d'entrée
class PredictionDataAppartement(BaseModel):
    surface: float
    nbRooms: float
    nbWindows: float
    price: float

# Initialisation du modèle de régression linéaire
model = LinearRegression()


# Initialisation du modèle de régression linéaire
modelSecond = LogisticRegression(max_iter=200)


# Initialisation du modèle de régression linéaire
modelThird = KNeighborsClassifier(n_neighbors=5)

# Utilisation cohérente de l'encodeur : L'encodeur label_encoder que vous avez utilisé pour transformer les catégories lors de l'entraînement doit être réutilisé pour inverser cette transformation lors de la prédiction.
label_encoder = LabelEncoder()



# Variable pour vérifier si le modèle est entraîné
is_model_trained = False

# Endpoint pour entraîner le modèle


@app.post("/train")
async def train():
    global is_model_trained

    # Lire le fichier CSV
    df = pd.read_csv('appartements.csv')

    # Extraction des variables indépendantes et dépendantes
    X = df[['surface']]  # Variable explicative (surface)
    y = df['price']  # Variable cible (prix)

    # Entraînement du modèle
    model.fit(X, y)
    
    bins = [0, 150000, 250000, 400000, float('inf')]  # Example thresholds
    labels = ['low', 'normal', 'high', 'scam']  # Classes
    df['price_category'] = pd.cut(df['price'], bins=bins, labels=labels)


    X = df[['nbRooms', 'surface', 'nbWindows', 'price']]  # Features
    y = df['price_category']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    


    # Train the model
    modelSecond.fit(X_train, y_train)

    # Marquer le modèle comme entraîné
    is_model_trained = True

    # Logging avec Loguru
    logger.info("Modèle entraîné avec succès.")
    logger.info(f"Coefficients: {model.coef_}, Intercept: {model.intercept_}")

    def classify_apartment_by_surface(surface):
        if surface < 40:
            return 'F1'
        elif 40 <= surface < 60:
            return 'F2'
        elif 60 <= surface < 80:
            return 'F3'
        else:
            return 'F4'

    df['apartment_type'] = df['surface'].apply(classify_apartment_by_surface)

    # Encodage des catégories F1, F2, F3, F4
    df['apartment_type_encoded'] = label_encoder.fit_transform(
        df['apartment_type'])

    # Extraction des variables indépendantes (surface et prix) et dépendante (type d'appartement)
    X = df[['surface',]]  # Variables explicatives
    y = df['apartment_type_encoded']  # Variable cible (F1, F2, F3, F4)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    modelThird.fit(X_train, y_train)


    return {"message": "Modèle entraîné avec succès."}

# Endpoint pour prédire un prix en fonction des données d'entrée


@app.post("/predict")
async def predict(data: PredictionData):
    global is_model_trained

    # Vérifier si le modèle a été entraîné
    if not is_model_trained:
        raise HTTPException(
            status_code=400, detail="Le modèle n'est pas encore entraîné. Veuillez entraîner le modèle d'abord.")

    X_new = np.array([[data.surface]])

    # Prédire le prix
    predicted_price = model.predict(X_new)[0]

    # Logging avec Loguru
    logger.info(f"Prédiction faite pour surface: {data.surface}")
    logger.info(f"Prix prédit: {predicted_price}")

    return {"predicted_price": predicted_price}


@app.post("/predict-category")
async def predictcategory(data: PredictionDataAppartement):
    X_new = np.array([[data.surface, data.nbRooms, data.nbWindows, data.price],])

    # Prédire le prix
    predicted_price = modelSecond.predict(X_new)[0]

    # Logging avec Loguru
    logger.info(f"Prédiction faite pour surface: {data.surface}")
    logger.info(f"Prix prédit: {predicted_price}")

    return {"predicted_price": predicted_price}


@app.post("/predict-type")
async def predictype(data: PredictionData):
    X_new = np.array(
        [[data.surface],])

  
    predicted_type_encoded = modelThird.predict(X_new)[0]
    predicted_type = label_encoder.inverse_transform(
        [predicted_type_encoded])[0]

    return {"predicted_type": predicted_type}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
