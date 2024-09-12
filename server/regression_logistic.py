import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Charger le fichier CSV dans un DataFrame pandas
df = pd.read_csv('appartements.csv')

# Créer des catégories à partir des prix
# Par exemple : prix faible < 90k, moyen entre 90k et 120k, élevé > 120k


def categorize_price(price):
    if price < 90000:
        return 'faible'
    elif 90000 <= price <= 120000:
        return 'moyen'
    else:
        return 'élevé'


# Appliquer la fonction pour créer une nouvelle colonne "price_category"
df['price_category'] = df['price'].apply(categorize_price)

# Afficher les premières lignes pour vérifier
print(df[['price', 'price_category']].head())

# Séparer les caractéristiques (features) et la nouvelle cible (target)
X = df[['nbRooms', 'surface', 'nbWindows']]  # Variables indépendantes
y = df['price_category']  # Variable dépendante (la catégorie de prix)

# Encoder les labels pour la régression logistique (transformation en 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Créer un modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle avec les données d'entraînement
model.fit(X_train, y_train)

# Prédire les catégories de prix pour les données de test
predictions = model.predict(X_test)

# Afficher les prédictions et les valeurs réelles
print("Prédictions :", le.inverse_transform(predictions))
print("Valeurs réelles :", le.inverse_transform(y_test))

# Évaluer le modèle
accuracy = model.score(X_test, y_test)
print(f"Précision du modèle : {accuracy*100:.2f}%")


# Prédire la catégorie de prix pour un nouvel appartement
# Exemple d'un nouvel appartement : 3 pièces, 14 m², 2 fenêtres
new_apartment = [[3, 14, 2]]  # Caractéristiques du nouvel appartement

# Prédire la catégorie
predicted_category = model.predict(new_apartment)

# Transformer l'encodage en catégorie réelle (faible, moyen, élevé)
predicted_category_label = le.inverse_transform(predicted_category)

print(f"Le nouvel appartement avec 3 pièces, 14 m² et 2 fenêtres est dans la catégorie : {predicted_category_label[0]}")
