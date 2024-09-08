from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP

from flask import Flask,request,jsonify
app = Flask(__name__)

# Load your documents and build the index
def load_documents_and_create_index(folder_path):
    documents = SimpleDirectoryReader(folder_path).load_data()
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Your embedding model
    return VectorStoreIndex.from_documents(documents, embed_model=embed_model)

DOCUMENT_FOLDER = 'documents/'
index = load_documents_and_create_index(DOCUMENT_FOLDER)


def predict(query):
    llm = LlamaCPP(
        # you can set the path to a pre-downloaded model instead of model_url
        model_path="./Main-Model-7.2B-Q5_K_M.gguf",
    )

    # Load documents from the folder and create the index (RAG setup)
    #DOCUMENT_FOLDER = 'documents/' # Replace with your folder path

    # Initialize the index at startup
   # index = load_documents_and_create_index(DOCUMENT_FOLDER)

    # Set up the query engine and run the query
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query)

    # Return the model's response
    return response


# query="what this document is all about"
# query_engine = index.as_query_engine()
# response = query_engine.query(query)
# print(response,end='',flush=True)
response = predict("what is the benefits of this idea in simple language")
print(response)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data["query"]
    print(user_query)
    ans = predict(user_query)
    print(ans)
    return jsonify({"ans": ans})

if __name__ == '__main__':
    app.run(debug=True)