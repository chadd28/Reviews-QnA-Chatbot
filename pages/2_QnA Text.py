import streamlit as st
import pandas as pd
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

st.title('QnA')
st.markdown('This chatbot can read pure text')

st.header('Upload a file')

OPEN_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')
st.sidebar.markdown("Where's My OpenAI API Key? https://platform.openai.com/api-keys")
ACTIVELOOP_TOKEN = st.sidebar.text_input('ActiveLoop Token', type='password')
st.sidebar.markdown("Where's My ActiveLoop Token? https://app.activeloop.ai/")
ACTIVELOOP_DATASET = st.sidebar.text_input('ActiveLoop Dataset Name', value="text_embedding_reviews")
st.sidebar.markdown("Change name if you run into errors.")

uploaded_file = st.file_uploader('Upload the csv file', type=['csv'])

# this prevents this section from running again everytime the user chats.
@st.cache_data
def load_openai_deeplake(df):
    if uploaded_file is not None:

        os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
        os.environ["ACTIVELOOP_TOKEN"] = ACTIVELOOP_TOKEN
        reviewsText = df['review'].tolist()

        #st.toast('Initializing OpenAIEmbeddings and Deep Lake vector database (10s)')          - had to comment these out cuz it was causing error
            
        # Initialize OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

        # Specify your organization ID and dataset name
        my_activeloop_org_id = "applegpt2023"
        my_activeloop_dataset_name = ACTIVELOOP_DATASET
        dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

        # Initialize Deep Lake vector database
        db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

        #st.toast('Initializing RecursiveCharacterTextSplitter (15s)')
        # Initialize RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        all_texts, all_metadatas = [], []

        # Loop through each document
        for i, reviewsText in enumerate(reviewsText):
            chunks = text_splitter.split_text(reviewsText)
            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append({"source": f"Review_{i+1}_{chunk}"})

        # Add chunks and metadata to Deep Lake
        db.add_texts(all_texts, all_metadatas)

        # Initialize OpenAI language model
        llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Create RetrievalQAWithSourcesChain
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                                chain_type="stuff",
                                                                retriever=db.as_retriever())
        st.success('Done!', icon="✅")
        return(chain, all_metadatas)


            
if uploaded_file is not None:
    st.markdown('## DataFrame Preview:')
    df = pd.read_csv(uploaded_file)
    st.markdown(f"Number of reviews: {len(df)}")
    st.write(df.head())   
    chain, all_metadatas = load_openai_deeplake(df) 

st.title("Yelp Restaurant QnA Bot")
st.markdown("Powered by ChatGPT 3.5 Turbo & Deeplake")
# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Bot: Hello! First upload your dataset, the enter your question below"})


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input("Enter Question"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    d_response = chain({"Context":"only use info from retriever in the chain", "question": prompt})


    # print the answer and sources for debugging
    print(d_response)


    # gets the sources (DISABLED I CANT GET IT TO PRINT THE WHOLE REVIEW)
    # sourcesResponse = []
    # for source in d_response["sources"].split(", "):
    #     sourcesResponse.append(source)

    answer = d_response["answer"]
    response = f'''Bot: {answer}'''

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
