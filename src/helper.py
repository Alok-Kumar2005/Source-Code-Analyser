import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language , RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings



def repo_ingestion(repo_url):
    os.makedirs("repo" , exist_ok= True)
    repo_path = "repo/"
    Repo.clone_from(repo_url , to_path= repo_path)


## loading repositores as documents

def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path , 
                                       glob = "**/*",
                                       suffixes=[".py"],
                                       parser = LanguageParser(language = Language.PYTHON , parser_threshold=500))
    
    documets = loader.load()
    return documets

## creating text chunks
def text_splitter(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,
                                                                      chunk_size = 200 , 
                                                                      chunk_overlap = 20)

## loading embeddings model
def load_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings