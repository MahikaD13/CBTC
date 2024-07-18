import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_text(text):
    text=text.lower()      #text lower
    text=nltk.word_tokenize(text)     #tokenization of text
    
    y=[]
    for i in text:
        if i.isalnum():    #only fetches alphanumeric character
            y.append(i)
        
    text=y[:]
    y.clear()
    for i in text:     #removes stopwords and punctuations from text
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:   #apply stemming on text
        y.append(ps.stem(i))
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
page_bg_image="""
             <style>
               background-color: #FFA07A
             </style>
             """
st.markdown(page_bg_image,unsafe_allow_html=True)

st.title("Email Spam Classifier")
input_email=st.text_area("Enter the message")

if st.button('Predict'):

    transformed_email=transform_text(input_email)
 
    vector_input=tfidf.transform([transformed_email])

    result=model.predict(vector_input)[0]

    if result==1:
       st.header("Spam")
    else:
       st.balloons()
       st.header("Not Spam")
