from distutils.log import debug

#Là nous importons les libraries principale pour notre application
from flask import Flask,request,jsonify, render_template, redirect, url_for
#Nous importons sklearn car celui que nous avons utiliser pour dévélopper le modèle
import sklearn 
#Nous importons notre algorithme
from sklearn.svm import SVC
#la librairie qui nous aider pour enregistrer notre modèle
import pickle
#numpy va nous aider pour les colones du modèle
import numpy as np

app=Flask(__name__) #Là nous préparons l'environnement


#Nous allons créer un dictionnaire pour donner des valeurs à notre target
model=pickle.load(open('ModelGlas.pkl','rb'))
dict_classe_lesion={
	1:"Glass lai",
    2:'Glass Ju',
    3:'Glass eau',
    4:'Glass colora',
    5:'Glass sucre',
    6:'Glass rouge',
    7:'null'
}

@app.route('/') #Nous spécifions la route
def  home(): #Nous allons définir une fonction
    return render_template('index.html') #Ici nous spécifions la route de notre page associé

@app.route('/predict',methods=['POST'])

def predict():
    models=pickle.load(open('ModelGlas.pkl','rb'))
    #comme nos données sont du type float, nous les convertissons en int
    int_features=[float(i) for i in request.form.values()]
    dernier_features=[np.array(int_features)]

    dernier_features=np.array([dernier_features]).reshape(1,9)
    predire=models.predict(dernier_features)
   #Nous rétranchons le target
    pred_class=predire.argmax(axis=-1)
    
    #Pour notre dictionnaire
    prediction=dict_classe_lesion[predire[0]]

    result=str(prediction)
    return render_template('index.html',prediction_text_='Votre type est: {}'.format(result))
    
    #Nous ajoutons une condition qui va vérifier l'application pour exécuter
if __name__=="__main__":
    app.run(debug=True) 