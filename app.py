from langchain.vectorstores import Chroma
from src.helper import *
from dotenv import load_dotenv
import os
from flask import Flask , request ,render_template , jsonify
from langchain_groq import ChatGroq
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

app = Flask(__name__)

load_dotenv()

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768" # or "llama2-70b-4096"
)

embeddings = load_embeddings()
persist_directory = 'db'
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


memory = ConversationSummaryMemory(llm = llm , memory_key="chat_history" , return_messages=True)

qa = ConversationalRetrievalChain.from_llm(llm , retriever=vectordb.as_retriever(search_type = "mmr" , search_kwargs = {"k":8}) , memory = memory)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input) })




@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)


