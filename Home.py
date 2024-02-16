import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Restaurant Reviews",
)

st.write("# Restaurant Reviews")
st.sidebar.success("Select a page above.")

st.write("""
        **Instructions:**
         
        1. Scrape the data of the Yelp website and download the csv file on the "Scrape Data" page.
        2. Enter your OpenAI API key and ActiveLoop token on the "QnA" page.
        3. Upload your output.csv file.
        4. Wait for the chat bot to load (20-60s)
        5. Ask questions relating to the reviews.
         """)
