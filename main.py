import streamlit as st
import pickle

tfdif= pickle.load(open('vectorizer_pkl','rb'))
model_LR= pickle.load(open('model_pkl','rb'))

st.title("News Classifier Ml Application")
st.subheader("Prediction")
news_text=st.text_area("Enter text","Type here")
prediction_Labels={'business':0,'entertainment':1,'sport':2,'politics':3,'tech':4}
if st.button("Classify"):
    st.text("Original text ::\n{}".format(news_text))
    vect_text= tfdif.transform([news_text]).toarray()
    prediction= model_LR.predict(vect_text)
    if prediction==0:
        st.write("business")
    elif prediction==1:
        st.write("entertainment")
    elif prediction==2:
        st.write("sport")
    elif prediction==3:
        st.write("politics")
    else:
        st.write("tech")