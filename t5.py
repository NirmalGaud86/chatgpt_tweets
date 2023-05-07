#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the saved model
model = joblib.load("nb_model.joblib")

# Streamlit app code
st.title("Chatgpt Tweets Sentiment Analysis App")

st.header("Enter the tweet here:")
# Input text from the user
user_input = st.text_area("", height=100)

# Create a predict button
if st.button("Predict"):
    # Preprocess the input text using the loaded CountVectorizer
    text_dtm = model['vect'].transform([user_input])

    # Make predictions
    prediction = model['nb'].predict(text_dtm)

    st.header("Prediction")
    # Display the predicted sentiment
    if prediction == 0:
        st.subheader("Negative sentiment")
    elif prediction == 1:
        st.subheader("Neutral sentiment")
    elif prediction == 2:
        st.subheader("Positive sentiment")

    # Generate word cloud for the input text
    words = model['vect'].get_feature_names()
    word_freq = text_dtm.toarray()[0]
    word_cloud_data = dict(zip(words, word_freq))

    wc = WordCloud(background_color='white')
    wc.generate_from_frequencies(word_cloud_data)

    st.header("Word Cloud")
    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


# In[ ]:




