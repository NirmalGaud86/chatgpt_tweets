#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the saved model
with open("nb_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app code
st.title("Chatgpt Tweets Sentiment Analysis App")

# Input text from the user
user_input = st.text_area("Enter your tweet here:")

# Create a predict button
if st.button("Predict"):
    # Preprocess the input text using the loaded CountVectorizer
    text_dtm = model['vect'].transform([user_input])

    # Make predictions
    prediction = model['nb'].predict(text_dtm)

    # Display the predicted sentiment
    if prediction == 0:
        st.write("Negative sentiment")
    elif prediction == 1:
        st.write("Neutral sentiment")
    elif prediction == 2:
        st.write("Positive sentiment")

    # Generate word cloud for the input text
    words = model['vect'].get_feature_names()
    word_freq = text_dtm.toarray()[0]
    word_cloud_data = dict(zip(words, word_freq))

    wc = WordCloud(background_color='white')
    wc.generate_from_frequencies(word_cloud_data)

    # Display the word cloud
    st.subheader("Word Cloud")
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


# In[ ]:




