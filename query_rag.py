
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from load_docs import load_docs, split_docs, embed_func

PROMPT_TEMPLATE = """
You are a helpful assistant. Use the provided context to answer the question.
Question: {question}
Context: {context}
Answer:
"""
def get_context(question):
    context_text = ""

    # retrieve the vector store
    vector_db = Chroma(
        persist_directory="Chroma",
        embedding_function=embed_func()
    )

    # query the vector store for relevant documents
    results = vector_db.similarity_search(question, k=2)
    for doc in results:
        context_text += doc.page_content + "\n"
    
    return context_text, results

def generate_prompt(question, context):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt.format(question=question, context=context)

def query_rag(prompt, results):
    # invoke LLM with the prompt
    sources = ""
    model = OllamaLLM(model='llama3.2')
    response = model.invoke(prompt)

    # sources 
    for doc in results:
        sources += f"\n{doc.metadata['source']}"
    
    format_response = f"{response}\n\nSources: {sources}"

    return format_response

def main():

    question = input("Ask a question: ")
    context, results = get_context(question)
    prompt = generate_prompt(question, context)
    response = query_rag(prompt, results)
    print(response)


if __name__ == "__main__":
    main()