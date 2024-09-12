<template>
  <div>
   <v-card class="mx-auto" max-width="600">
    <v-card-title>
      <h3>Ajouter un nouvel appartement</h3>
    </v-card-title>
    <v-card-text>
    <form @submit.prevent="ajouterAppartement">
      <div>
        <v-text-field label="Nombre de chambres" v-model="nouvelAppartement.nbRooms" type="number" id="nbRooms" required />
      </div>
      <div>
        <label for="surface">Surface (m²) :</label>
        <v-text-field v-model="nouvelAppartement.surface" type="number" id="surface" required />
      </div>
      <div>
        <label for="nbWindows">Nombre de fenêtres :</label>
        <v-text-field v-model="nouvelAppartement.nbWindows" type="number" id="nbWindows" required />
      </div>
      <div>
        <label for="price">Prix :</label>
        <v-text-field v-model="nouvelAppartement.price" type="number" id="price" required />
      </div>
      <v-btn type="submit">Ajouter l'appartement</v-btn>
    </form>
      </v-card-text>
  </v-card>
  </div>
</template>

<script>
import { defineComponent, reactive } from 'vue';

export default defineComponent({
  name: "AjoutAppartement",
  emits: ['appartement-ajoute'],
  setup(props, { emit }) {
    const nouvelAppartement = reactive({
      nbRooms: 4,
      surface: 20,
      nbWindows: 3,
      price: 100000
    });

    const ajouterAppartement = () => {
      // Générer un ID unique pour le nouvel appartement
      const nouvelId = Date.now();
      const appartementComplet = { ...nouvelAppartement, id: nouvelId };
      emit('appartement-ajoute', appartementComplet);

      // Réinitialiser le formulaire
      nouvelAppartement.nbRooms = 0;
      nouvelAppartement.surface = 0;
      nouvelAppartement.nbWindows = 0;
      nouvelAppartement.price = 0;
    };

    return { nouvelAppartement, ajouterAppartement };
  }
});
</script>