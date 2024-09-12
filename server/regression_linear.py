import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Charger le fichier CSV dans un DataFrame pandas
df = pd.read_csv('appartements.csv')

# Visualisation des premières lignes du fichier pour vérifier que les données sont bien chargées
print(df.head())

# Séparer les caractéristiques (features) et la cible (target)
X = df[['nbRooms', 'surface', 'nbWindows']]  # Variables indépendantes
y = df['price']  # Variable dépendante (le prix)

# Diviser les données en ensembles d'entraînement et de test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Créer un modèle de régression linéaire
model = LinearRegression()

# Entraîner le modèle avec les données d'entraînement
model.fit(X_train, y_train)

# Prédire les prix pour les données de test
predictions = model.predict(X_test)

# Prédire le prix d'un nouvel appartement
# Par exemple : un appartement avec 3 pièces, 14 m² de surface et 2 fenêtres
new_apartment = [[3, 14, 2]]  # Nouvelle entrée pour prédiction

predicted_price = model.predict(new_apartment)

print(f"Le prix prédit pour l'appartement avec 3 pièces, 14 m² et 2 fenêtres est : {predicted_price[0]:.2f} €")

# Afficher les prédictions et les prix réels
# print("Prédictions :", predictions)
# print("Prix réels :", y_test.values)

# Calculer la précision du modèle
# score = model.score(X_test, y_test)
# print(f"Précision du modèle : {score*100:.2f}%")
