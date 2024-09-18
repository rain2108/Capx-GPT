import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.twitter import TwitterTweetReader

st.set_page_config(
    page_title="YO",
    page_icon="👋",
)
logo = "C:/Users/91823/Desktop/llamaindex/capxgpt.png"
st.image(logo, width=400)
st.title('CapX guide for 🫵')


key = st.secrets["groq_api_key"]
BEARER_TOKEN = st.secrets["bearer_token"]

urls = [
        "https://mirror.xyz/capxai.eth/A7mtpiIwR7gxWFKy9yuVVnss7Pym0OwAXRpweNEGqbA",
        "https://mirror.xyz/capxai.eth/dyKdsq-FiiJ3oxh-kSNBFWitYVv4JUD_SXaQPtmCCKg",
        "https://mirror.xyz/capxai.eth/MdsOkZm1ydludd1B1W-n5ElQuMnMuy1RQsh8Fcxkefw",
        "https://mirror.xyz/capxai.eth/nIXYmEGh__6M5u19d-vENaqcq8iOyXC05zipQ9dw6P0",
        "https://mirror.xyz/capxai.eth/Sx31DySk3ZlcBTYFAPaMfeNyx4ELVaQdxzSZcyvXodM"
        ]

llm = Groq(model="llama3-70b-8192", api_key=key)


loader = BeautifulSoupWebReader()
docs = loader.load_data(urls=urls)

# x_reader = TwitterTweetReader(BEARER_TOKEN)
# # x_handles = ["@0xCapx", "@0xHBx", "@anshitaksoni", "@varishbajaj"]
# x_docs = x_reader.load_data(twitterhandles=["0xCapx"])

all_docs = docs #+ x_docs

Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_documents(all_docs)
query_engine = index.as_query_engine(llm=llm)

question = st.text_input("Ask anything about Capx here 👇:")

if st.button("Get Answer"):
    response = query_engine.query(question)
    st.write(response.response)

