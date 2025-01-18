import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, precision_recall_curve
import openpyxl
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

data = pd.read_csv("risk_credit.csv",sep=",",decimal=".")

df = data[data['person_age']<70]

st.title("Application de Machine Learning sur l'évaluation de la probabilité de défaut de paiement des emprunteurs.")
st.subheader("Auteur: Mamadou DIOP")

st.sidebar.title("Sommaire")

pages = ["Présentation du projet", "Exploration des données", "Visualisation des données", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    
    st.write("### Présentation du projet")
    st.write("Contexte :")
    
    st.write("Dans le domaine bancaire et assurantiel, l'évaluation précise du risque de crédit est cruciale pour minimiser les pertes et optimiser la gestion des portefeuilles de prêts.")
    
    st.write("Les institutions financières utilisent des modèles de risque de crédit pour déterminer la solvabilité des clients potentiels et pour établir les conditions de crédit appropriées.")
    
    st.write("Ce projet s'inscrit dans cette logique en fournissant un outil interactif et intuitif pour explorer et modéliser les données de crédit.")
    
    st.write("Objectifs :")
    
    st.write("L'objectif principal de ce projet est de développer une application web interactive pour analyser et modéliser les risques de crédit à l'aide de techniques de science des données.")
    
    st.write("Cette application permettra aux utilisateurs d'explorer un jeu de données de crédit, de visualiser diverses analyses graphiques, et de construire des modèles prédictifs pour évaluer la probabilité de défaut de paiement des emprunteurs.")
    
    
elif page == pages[1]:
    st.write("### Exploration des données")
    
    st.dataframe(df.head(10))

    st.write("Nombre de lignes : ", df.shape[0])   
    st.write("Nombre de colonnes : ", df.shape[1])  
    if st.checkbox("info sur la base"):
         st.write(df.info())
    if st.checkbox("Afficher les statistiques descriptives"):
         st.write(df.describe().T)
    if st.checkbox("Afficher les statistiques descriptives pour les variables catégorielles"):
         f1 = df.select_dtypes('object')
         st.write(f1.describe().T)     

    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
    if st.checkbox("Afficher les données manquantes aprés avoir remplacer les valeurs NaN par mode/médiane") : 
        df['person_emp_length'].fillna(df['person_emp_length'].mode()[0], inplace=True)
        df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
        st.write(df.isna().sum())
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())


elif page == pages[2]:

    st.write("### Visualisation des données")
    fig0 = sns.displot(x='loan_status', data=df, kde=True)
    plt.title("Distribution de la variable cible loan_status")
    st.pyplot(fig0)

    if st.checkbox("Afficher la relation entre deux variables"):
            var1 = st.selectbox("Sélectionnez la première variable", data.columns)
            var2 = st.selectbox("Sélectionnez la deuxième variable", data.columns)
            fig, ax = plt.subplots()
            sns.scatterplot(x=var1, y=var2, data=data, ax=ax)
            st.pyplot(fig)

    if st.checkbox("Afficher la distribution des variables catégorielles"):
       variable = st.selectbox("Sélectionnez une variable catégorielle", df.select_dtypes(include=['object']).columns)
       fig, ax = plt.subplots()
       sns.countplot(x=variable, data=df, ax=ax)
       st.pyplot(fig)
    
    if st.checkbox("Afficher la matrice de corrélation"):
            df_numeric = df.select_dtypes(include=[float, int])
            fig, ax = plt.subplots()
            sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
            plt.title("Matrice de corrélation des variables du dataframe")
            st.pyplot(fig)
        
    if st.checkbox("Afficher les histogrammes"):
            variable = st.selectbox("Sélectionnez une variable", data.columns)
            fig, ax = plt.subplots()
            sns.histplot(data[variable], kde=True, ax=ax)
            st.pyplot(fig)

elif page == pages[3]:
    st.write("### Modélisation")
    
    # Ajoutez une option de sélection pour le type de modèle
    model_type = st.selectbox("Choisissez le type de modèle", ["SVM", "Random Forest", "Régression Logistique"])

    y = df["loan_status"]
    X = df.drop("loan_status", axis=1) 

    if st.button("Entraîner le modèle"):
       # Encodage des variables catégorielles
       label_encoders = {}
       for col in X.columns:
           if X[col].dtype == 'object':
              label_encoders[col] = LabelEncoder()
              X[col] = label_encoders[col].fit_transform(X[col])
        
       X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=123)
       imputer = SimpleImputer(strategy='mean')
       X_train = imputer.fit_transform(X_train)
       X_test = imputer.transform(X_test)
       # Sélectionnez le modèle en fonction de l'option choisie
       if model_type == "SVM":
          model = SVC()
       elif model_type == "Random Forest":
          model = RandomForestClassifier()
       elif model_type == "Régression Logistique":
           model = LogisticRegression()
       #Entrainement du model
       model.fit(X_train, y_train)

      #Prédictions
       predictions = model.predict(X_test)

      #Métriques de performance
       accuracy = model.score(X_test,y_test)
       precision = precision_score(y_test,predictions)
       recall = recall_score(y_test,predictions)

       # Afficher les métriques dans l'application
       st.write("Accuracy: {:.2f}".format(accuracy))
       st.write("Precision: {:.2f}".format(precision))
       st.write("Recall: {:.2f}".format(recall))

       st.write("Classification Report")
       st.text(classification_report(y_test, predictions))
      
