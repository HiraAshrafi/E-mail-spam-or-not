import streamlit as st
import numpy as np
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()









def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    
    
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
        

tfid=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open("mnb_model.pkl","rb"))


st.title("Email/sms spam classifier prediction ")



input_sms=st.text_area("Enter The Message")

if st.button("Predict"):
    transformed_sms=transform_text(input_sms)
    vector_input=tfid.transform([transformed_sms])
    result=model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
    
    
    
    
    
    
    

 

    
    

