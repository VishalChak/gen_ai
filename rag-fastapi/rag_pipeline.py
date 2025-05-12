from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.groq import Groq


class RAGPipeline:
    def __init__(self, model, api_key, embed_model, data_dir):
        self.data_dir = data_dir
        self.embed_model = embed_model
        self.llm = Groq(
            model = model,
            api_key = api_key
        )
        self.index = None
        self.query_engine = None
        self._setup()


    def _setup(self):
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        self.index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.index.as_query_engine(
            similirity_top_k = 3,
            response_mode = "compact"
        )

    def response(self, question):
        return self.query_engine.query(question)


# if __name__ == "__main__":
#     groq_api_key = ""
#     model = "compound-beta"
#     embed_model = "local:BAAI/bge-small-en-v1.5"
#     data_dir = "data"

#     rag = RAGPipeline(model = model, api_key = groq_api_key , embed_model = embed_model, data_dir = data_dir)
#     print(rag.response("Who is vishal"))