from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP

from flask import Flask,request,jsonify
app = Flask(__name__)

global index

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Your embedding model

llm = LlamaCPP(
    # you can set the path to a pre-downloaded model instead of model_url
    model_path="./Main-Model-7.2B-Q5_K_M.gguf",
)

@app.route("/recreateVectorStoreIndex")
# Load your documents and build the index
def load_documents_and_create_index():
    documents = SimpleDirectoryReader('documents/').load_data()
    global index
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    return jsonify({"success":"success"})

def predict(question):
    # Set up the query engine and run the query
    global index
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(question)

    # Return the model's response
    return response


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data["query"]
    ans = predict(user_query)
    return jsonify({"ans": str(ans)})

if __name__ == '__main__':
    app.run(debug=True)