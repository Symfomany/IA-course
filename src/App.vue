<template>
 <v-app>
    <!-- Barre de navigation ou entête -->
    <v-app-bar app color="primary" dark>
      <v-toolbar-title>Gestion des Appartements</v-toolbar-title>
    </v-app-bar>

    <!-- Contenu principal -->
    <v-main>
      <v-container class="mt-5">
        <Listing />
      </v-container>
    </v-main>

    <!-- Pied de page -->
    <v-footer app color="primary" dark>
      <v-col class="text-center" cols="12">
        © 2024 - Mon Application de gestion d'appartements
      </v-col>
    </v-footer>
  </v-app>
</template>

<script>
import Listing from './components/Listing.vue';

export default {
  name: 'App',

  components: {
    Listing,
  },

  data: () => ({
    //
  }),
}
</script>
