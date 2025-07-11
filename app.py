import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from PIL import Image

logo = Image.open("assets/logo.jpg")

@st.cache_data
def load_data():
    return pd.read_csv("data/housing.csv")

df = load_data()

st.image(logo, width=150)

st.title(" Analyse des Prix des Logements en Californie")
st.markdown("Par * Mohammed El Kima  – Été 2025 – Institut Teccart**")

page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Analyse descriptive", "Visualisations", "Corrélations", "Recommandations","Rapport final"]
)

if page == "Rapport final":
    st.header(" Rapport final : Analyse des facteurs influençant le prix des maisons")
    st.write("""
Ce rapport présente une analyse exploratoire des données sur les logements en Californie. 
L’objectif est d’identifier les facteurs clés influençant les prix immobiliers et de formuler des 
recommandations à destination des agences immobilières pour optimiser leurs stratégies.
""")


    st.markdown("""
    ### 1. Facteurs majeurs impactant le prix des maisons
    Après analyse statistique et modélisation par régression linéaire, voici les facteurs qui ont le plus grand impact sur le prix des logements en Californie :

    - **Median Income (Revenu médian)** : Le facteur le plus influent. Les zones avec un revenu médian plus élevé ont des prix immobiliers plus élevés.
    - **Longitude et Latitude (Localisation)** : La position géographique, notamment la proximité des côtes et des centres urbains, impacte fortement les prix.
    - **Total Rooms & Total Bedrooms** : Plus une maison a de pièces, plus sa valeur tend à augmenter, sous réserve de la qualité et surface.
    - **Housing Median Age (Âge médian des logements)** : L’ancienneté des maisons influence modérément la valeur, les logements plus récents étant souvent plus chers.

    ### 2. Recommandations pour les agences immobilières

    - **Cibler les zones à revenu médian élevé** : Investir dans les quartiers où le revenu médian est élevé pour maximiser la rentabilité.
    - **Mettre en avant la localisation** : Valoriser les biens proches des zones côtières et des pôles urbains attractifs.
    - **Valoriser le nombre de pièces et la surface** : Promouvoir les logements avec un nombre de pièces conséquent et bien proportionnées.
    - **Prendre en compte l’état et l’âge des logements** : Recommander des rénovations pour les maisons plus anciennes afin d’augmenter leur valeur.
    - **Utiliser des analyses de données régulièrement** : Intégrer des outils analytiques pour suivre l’évolution des prix et adapter les stratégies en temps réel.

    ### 3. Ressources

    - Code source complet et notebook associé disponibles sur GitHub :  
      [https://github.com/tonpseudo/immobilier-californie](https://github.com/tonpseudo/immobilier-californie)

    """)



if page == "Accueil":
    st.subheader("Bienvenue ! ")
    st.write("""
    Cette application interactive permet d’explorer les facteurs influençant les prix des logements en Californie.
    """)

elif page == "Analyse descriptive":
    st.subheader(" Statistiques descriptives")
    st.dataframe(df.head())
    st.write(df.describe())
    st.write("Valeurs manquantes :")
    st.write(df.isnull().sum())

elif page == "Visualisations":
    st.subheader(" Histogramme des prix")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['median_house_value'], bins=50, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Boxplot âge vs prix")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['housing_median_age'], y=df['median_house_value'], ax=ax2)
    st.pyplot(fig2)

    st.subheader("Carte des prix selon localisation")
    fig3, ax3 = plt.subplots()
    scatter = ax3.scatter(df['longitude'], df['latitude'], c=df['median_house_value'], cmap="viridis", alpha=0.5)
    plt.colorbar(scatter, ax=ax3, label='Prix moyen')
    st.pyplot(fig3)

elif page == "Corrélations":
    st.subheader(" Corrélation avec le prix")
    corr = df.corr(numeric_only=True)["median_house_value"].sort_values(ascending=False)
    st.write(corr)

    st.subheader(" Importance via régression linéaire")
    df_clean = df.dropna()
    X = df_clean.drop("median_house_value", axis=1).select_dtypes(include=[np.number])
    y = df_clean["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    coeff = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
    st.write(coeff)

elif page == "Recommandations":
    st.subheader(" Recommandations stratégiques")
    st.markdown("""
    - Ciblez les zones à fort revenu médian** : elles sont fortement liées à des prix plus élevés.
    - Surveillez les zones côtières et urbaines** (latitude/longitude) : elles concentrent les logements les plus chers.
    - Considérez l'âge et l'état des maisons** : l’ancienneté influence modérément le prix.
    - Investissez dans des logements avec plus de pièces, mais proportionnés à la surface.**
    -Cibler les zones à revenu élevé
    -Investir dans les quartiers où le revenu médian dépasse 4.0 pour viser des biens à forte valeur.
    -Mettre en avant la localisation
    Les biens proches des côtes et des grandes villes (latitude > 34, longitude < -120) ont plus de valeur.

...

    """)
    

