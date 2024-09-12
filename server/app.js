const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const json2csv = require('json-2-csv');  

const app = express();
const PORT = 3000;
const cors = require('cors');

app.use(cors())

// Middleware pour traiter les requêtes JSON
app.use(bodyParser.json());

// Chemin vers le fichier JSON où les appartements sont stockés
const filePath = path.join(__dirname, 'appartements.json');

// Chemin vers le fichier CSV où les appartements seront convertis
const csvFilePath = path.join(__dirname, 'appartements.csv');

// Fonction utilitaire pour lire les appartements depuis le fichier JSON
const readAppartementsFromFile = () => {
  try {
    const data = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(data);
  } catch (err) {
    return []; // Si le fichier n'existe pas ou est vide, retourner un tableau vide
  }
};

// Fonction pour convertir le fichier JSON en CSV et l'écrire dans un fichier
const convertJsonToCsv = async (appartements) => {
     try {
        const csv = await json2csv.json2csv(appartements); // Utilisation de la version async de json2csv
        fs.writeFileSync(csvFilePath, csv, 'utf8'); // Écrire dans le fichier CSV
        console.log('Fichier CSV mis à jour avec succès');
    } catch (err) {
        console.error('Erreur lors de la conversion JSON vers CSV :', err);
    }
    
};

// Fonction utilitaire pour écrire les appartements dans le fichier JSON
const writeAppartementsToFile = async (appartements) => {
    fs.writeFileSync(filePath, JSON.stringify(appartements, null, 2), 'utf8');

    await convertJsonToCsv(appartements);
};

// Route pour récupérer la liste des appartements
app.get('/api/appartements', async (req, res) => {
  const appartements = readAppartementsFromFile();
  res.json(appartements);
});

// Route pour ajouter un nouvel appartement
app.post('/api/appartements', async (req, res) => {
  const newAppartement = req.body;

  // Lire les appartements existants
  const appartements = readAppartementsFromFile();

  // Ajouter l'ID unique (basé sur la date actuelle)
  newAppartement.id = Date.now();

  // Ajouter le nouvel appartement à la liste
  appartements.push(newAppartement);

  // Écrire la nouvelle liste dans le fichier JSON
   await writeAppartementsToFile(appartements);


  // Retourner l'appartement ajouté avec l'ID
  res.status(201).json(appartements);
});

// Démarrer le serveur
app.listen(PORT, () => {
  console.log(`Serveur en écoute sur http://localhost:${PORT}`);
});
