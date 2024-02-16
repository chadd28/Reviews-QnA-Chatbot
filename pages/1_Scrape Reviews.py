import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import time

st.title('Scrape data')

url = st.text_input("Yelp Restaurant URL")

if st.button("Scrape Reviews"):
    with st.status("Performing Webscrape...", expanded=True) as status:
        st.write("Scraping from + " + url + "...")
        # collecting the data
        root = url
        r = requests.get(root)
        print(r.status_code)
        soup=BeautifulSoup(r.text, 'html.parser')
        divs = soup.findAll(class_="comment__09f24__D0cxf css-qgunke")
        reviews = []
        for div in divs:
            reviews.append(div.find('span').text)

        # looping through pages of yelp restaurant to scrape each review
        for page in range (10, 80, 10):
            r = requests.get(root+'?start='+str(page))
            soup=BeautifulSoup(r.text, 'html.parser')
            divs = soup.findAll(class_="comment__09f24__D0cxf css-qgunke")
            for div in divs:
                reviews.append(div.find('span').text)
        # df = pd.read_csv('reviews.csv')         # used for testing when requests are overloaded
        # reviews = df['Review'].tolist()         # used for testing when requests are overloaded
        
        df = pd.DataFrame(np.array(reviews), columns=['review'])

        csv = df.to_csv(index=False)

        # shows dataframe and download button
        st.markdown(f"Number of reviews scraped: {len(df)}")
        st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='output.csv',
        mime='text/csv',)
        st.dataframe(df)
        status.update(label="Webscrape complete!")