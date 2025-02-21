# Importation des bibliothèques nécessaires
import streamlit as st  # Importer Streamlit pour l'application Web

# Bibliothèques externes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Configuration de l'affichage des graphiques dans Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)


df = pd.read_csv('iris.csv' ,delimiter=';')

# Afficher les premières lignes du jeu de données
print(df.head())
# Statistiques descriptives pour comprendre la distribution des caractéristiques
print(df.describe())
#Affichage de la liste des colonnes 
print("Voici la liste des colonne de notre jeu de données")
print(df.columns)

#Exercices : Visualisation des données d’iris
effectifs=df['species'].value_counts()
#Exercice 1 :
print("#########################################################")
print("Debut des exercices")
print("Question N°1: Affichage des effectifs des modalités")
print(effectifs)

# Séparer les caractéristiques et la cible
X = df.drop('species', axis=1)
y = df['species']
# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# Normaliser les caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Créer le modèle KNN
knn = KNeighborsClassifier(n_neighbors=3)
# Entraîner le modèle
knn.fit(X_train, y_train)
# Prédire les classes de l'ensemble de test
y_pred = knn.predict(X_test)
# Afficher la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=df['species'].unique(), yticklabels=df['species'].unique())
plt.title('Matrice de confusion')
plt.xlabel('Prédictions')
plt.ylabel('Vraies classes')
plt.show()
# Calculer l'exactitude
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitude du modèle : {accuracy * 100:.2f}%")
# Afficher le rapport de classification
print("Rapport de classification :\n", classification_report(y_test,y_pred))
#Affichage des differents diagrammes
print("2. Representation des données sous forme de diagramme")
print("2.a) Histogramme")
plt.bar(effectifs.index, effectifs.values, color=['blue','green','orange'])
plt.title("Effectif des trois especes d'iris")
plt.xlabel("Especes")
plt.ylabel("Effectifs")
plt.show()

#données representées en secteurs
print("2.b) Representation des données en secteurs")
plt.figure(figsize=(8,8))
effectifs.plot.pie(autopct='%1.1f%%',startangle=90, colors=['green','red','yellow'], labels=effectifs.index)
plt.ylabel('')
plt.show()

#Diagrammes à barres groupés
group_donnees=df.groupby('species').mean()
group_donnees.plot(kind='bar',figsize=(10,6))
plt.title('Moyenne des caracteristiques par espece')
plt.xlabel('Especes')
plt.ylabel('Valeur moyenne')
#Afficher la legende
plt.legend(title='Caracteristiques')

#Affichage
plt.tight_layout()
plt.show()

#calcul des moyennes des longueurs des sepales
moyenne_longueur_sepal=df.groupby('species')['SepalLength'].mean()

#Calcul des valeurs pour le diagramme en cascade
categories=moyenne_longueur_sepal.index
values=moyenne_longueur_sepal.values

#initialisation des valeurs cummulées
cumulative_values=[0]
#Calcul des valeurs cumulées
for i in range(1,len(values)):
    cumulative_values.append(cumulative_values[-1]+values[i])
    fig, ax=plt.subplots(figsize=(10,6))
for i in range(1,len(cumulative_values)):
    ax.bar(categories[i],values[i-1],bottom=cumulative_values[i-1],color='skyblue',edgecolor='grey')
#Ajout d'un titre et des labels
plt.title('Diagramme en cascade des moyennes de la longueur des sepales par espèce')
plt.xlabel('Especes')
plt.ylabel('Longueur moyenne des sepales')
plt.tight_layout()
plt.show()

#Selectionner les colonnes numeriques(exclure la colonne cible si presente)
features = df.select_dtypes(include=[np.number]).columns.tolist()




