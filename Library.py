from typing import Annotated, Literal, TypedDict
import os
import PyPDF2
import assemblyai as aai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader



from langchain_groq import ChatGroq
from langchain.schema import Document

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain


from langchain.chains import create_retrieval_chain

from langchain_core.runnables import RunnablePassthrough

from langchain_community.vectorstores import FAISS


from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from langchain_community.tools import DuckDuckGoSearchRun

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph,MessagesState, START, END
from IPython.display import Image, display

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import langchain_core
from typing import List
import langchain_community

from pprint import pprint

from langchain_community.tools.tavily_search import TavilySearchResults


from langgraph.checkpoint.memory import MemorySaver

from langchain_community.document_loaders import PyPDFLoader


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()



os.environ["ASSEMBLY_AI_KEY"]=os.getenv('ASSEMBLY_AI_KEY')
aai.settings.api_key = os.getenv("ASSEMBLY_AI_KEY")


GROQ_API_KEY=os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY']=GROQ_API_KEY


os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]= "Rag Tool for NLP"

os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')

os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

class LLMSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # cls._instance = ChatGroq(model_name="llama-3.1-70b-versatile")
            # 	DEVELOPER	CONTEXT WINDOW (TOKENS)	MAX OUTPUT TOKENS
            	# Meta	128k	32,768
            cls._instance = ChatGroq(model_name="llama-3.3-70b-versatile")

        return cls._instance
    
class GeminiSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # cls._instance = ChatGroq(model_name="llama-3.1-8b-instant")
            # cls._instance = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            cls._instance = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        return cls._instance
    

transcriber = aai.Transcriber()
def transcribe_audio_or_video(file_path):
    transcript = transcriber.transcribe(file_path)
    return transcript.text


def refine_transcript(unfiltered_text):
  llm = LLMSingleton.get_instance()
  system = """
  You are an expert in filtering text by removing unnecessary conversational phrases.
  Your task is to clean up the following transcript and remove all the filler content, such as phrases like "let me show you how", "I suppose you got my point", or any similar non-essential conversation.
  Keep only the relevant and meaningful content.Convert the text if need be such that the file text contains formal theory removing informal phrases and unnecessary phrases.
  Only return the filtered content and don't add any extra words or phrases from your end like:"here is the filtered content."
  The transcript is as follows:

  {transcript}
  """
  filter_text_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system),
          ("human", "The transcript: \n\n {transcript}"),
      ]
  )

  retrieval_grader = filter_text_prompt | llm | StrOutputParser()
  filtered_text = retrieval_grader.invoke({"transcript":unfiltered_text})
  return filtered_text


def text_extractor(file_path,file_type):
  # Extract text based on file type
  text_content = ""
  if file_type == "PDF":
    # Extract text from PDF using PyPDF2
    with open(file_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text_content += page.extract_text()  # Extract text from each page
  elif file_type == "Video":
      # Extract text from video using Assembly AI

    text_content = transcribe_audio_or_video(file_path)
  else:
    print("Unsupported file type")
  # Limit set to 5000 because we are using llama3.3
  text_content = refine_transcript(text_content[:5000])

  if len(text_content) > 5000:
    text_content += refine_transcript(text_content[5000:10000])

  return text_content




def store_text_in_vector_db(text):
    # Convert the input text into a document
    docs = [Document(page_content=text)]

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the document into smaller chunks
    documents = text_splitter.split_documents(docs)

    
    embeddings = GoogleGenerativeAIEmbeddings(
    # model="models/embedding-001",
    model="models/text-embedding-004",
    transport="rest"          # ← this forces the sync/REST client
)


    # Create a Chroma vector store from the documents and embeddings
    # db = Chroma.from_documents(documents, embeddings)
    db = FAISS.from_documents(documents, embeddings)


    # Perform a similarity search based on the input query
    # retrieved_results = db.similarity_search(query)

    # Return the retriever object for further use
    retriever = db.as_retriever()

    # Optionally, return retriever and results if needed for further use
    return retriever

def search_text_from_vector_db(retriever, query):
    # Perform a similarity search based on the input query
    retrieved_results = retriever.get_relevant_documents(query)

    text_content=""

    # Print the content of the most relevant document
    for result in retrieved_results:
        text_content += result.page_content
    return text_content


# Function to get the rag chain
def create_rag_pipeline(retriever,text,docs=None):
    # Assuming `hub.pull` retrieves the prompt template from a repository
    llm = LLMSingleton.get_instance()
    if not docs:
      docs = [Document(page_content=text)]

    prompt = hub.pull("rlm/rag-prompt")

    # Function to format the documents into a string format
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain with context and question, then process it with the LLM
    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    rag_chain = prompt | llm | StrOutputParser()


    # Run the chain and get the result
    # result = rag_chain.invoke(question)

    # return result
    return rag_chain


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def create_retrieval_grader(GradeDocuments):
    """Function to create and return the retrieval grader."""
    llm = LLMSingleton.get_instance()

    # Define the system prompt for grading
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    # Define the grade prompt template
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    # Initialize the structured LLM grader
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    retrieval_grader = grade_prompt | structured_llm_grader

    # Return the final retrieval grader with the grade prompt
    return retrieval_grader


def create_question_rewriter():
    """Function to create and return the question rewriter."""
    llm = LLMSingleton.get_instance()
    # Define the system prompt for rewriting the question
    system = """You will be given a question which is aimed to extract answer from some pdf.
    You have to convert the question in the format such that it seems,
    as if the user has intended to get the answer from the web and not from the pdf.
    Also just return the new question and no other verbose."""

    # Define the rewrite prompt template
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    # Create the question rewriter with the defined prompt and output parser
    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # Return the question rewriter
    return question_rewriter


class GradeQuestion(BaseModel):
    """Binary score for checking whether the question is technical or not."""

    binary_score: str = Field(
        description="Question is technical or not,if yes:'technical' else:'non-technical'"
    )


def create_question_grader():
  """Function to create and return the question grader."""
  llm = LLMSingleton.get_instance()
  llm_with_structured_output = llm.with_structured_output(GradeQuestion)
  # Define the system prompt for rewriting the question
  system = """You will be given a question. Assume you have access to a rag pipeline for complex technical concepts.
  Now your role is to classify the question whether the question should be forwarded to rag pipeline or should it be
  answered by your own knowledge.
  Eg,
  1)'Explain gravitational force', the output should be technical
  2)'Hi! How are you?', the output should be non-technical
  3)'Explain Attention Mechanism', the output should be technical
  4)'What is my name?', the output should be non-technical
  5) What is the meaning of <some technical term>, the output should be technical
  6) What is the meaning of <some non-technical term>, the output should be non-technical

  Note: things like art, cinema and politics are also technical terms.
  """

  # Define the rewrite prompt template
  re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the question: \n\n {question}",
        ),
    ]
  )

  # Create the question rewriter with the defined prompt and output parser
  question_grader = re_write_prompt | llm_with_structured_output

  # Return the question rewriter
  return question_grader

class State_Corrective_Rag(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        technical: String that represents whether the rag pipeline is to be used or
        if it is a general question.
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
    technical: str

def plot_graph(app):
    """Function to display a mermaid graph for a given app."""
    display(Image(app.get_graph().draw_mermaid_png()))



class Chatbot:
  def __init__(self):
    self.llm = LLMSingleton.get_instance()

  def set_tools(self,text):
    # self.retriever = store_text_in_vector_db(text)
    self.retrieval_grader = create_retrieval_grader(GradeDocuments)
    self.question_rewriter = create_question_rewriter()
    self.retrieval_chain = create_rag_pipeline(self.retriever,text)
    self.web_search_tool = TavilySearchResults(k=3)
    self.question_grader_llm = create_question_grader()
    # self.memory = MemorySaver()

  def question_grader(self,state):
    """
    Decides whether the question is meant for rag pipeline or for the bot for general qna.

    Args:
        state (dict): The current graph state

    Returns:

    """

    question = state["question"]

    question_grader_llm = self.question_grader_llm
    result = question_grader_llm.invoke({"question":question})

    grade = 'technical'

    if(result.binary_score == 'non-technical'):
      grade = 'non-technical'

    return {"technical":grade}

  def general_llm(self,state):
    """
    Provides answers on the basis of the llm's knowledge.

    Args:
        state (dict): The current graph state

    Returns:

    """
    question = state["question"]
    ans = self.llm.invoke(question)
    return {"generation":ans.content}

  def retrieve(self,state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:

    """

    question = state["question"]

    retriever = self.retriever

    # print("here...")
    documents = retriever.get_relevant_documents(question)

    return {"documents": documents, "question": question}

  def grade_documents(self,state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """


    question = state["question"]
    documents = state["documents"]
    # retrieval_grader = state["retrieval_grader"]
    retrieval_grader = self.retrieval_grader


    # Score each doc
    filtered_docs = []

    web_search = "No"

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


  def transform_query(self,state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    question = state["question"]

    documents = state["documents"]

    # question_rewriter = state["question_rewriter"]
    question_rewriter = self.question_rewriter

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})

    return {"documents": documents, "question": better_question}

  def web_search(self,state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    question = state["question"]

    documents = state["documents"]

    # web_search_tool = state["web_search_tool"]
    web_search_tool = self.web_search_tool
    # Web search
    docs = web_search_tool.invoke({"query": question})








    web_results = "\n".join([
    d["content"] for d in docs
    if isinstance(d, dict) and "content" in d
])




    result = self.llm.invoke(question)

    # print("web_results...",web_results)
    web_results = Document(page_content=web_results)


    documents.append(web_results)

    return {"documents": documents, "question": question}




  def generate(self,state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    # print("---GENERATE---")

    question = state["question"]
    documents = state["documents"]
    # retrieval_chain = state["retrieval_chain"]
    retrieval_chain = self.retrieval_chain
    # print("DOCUMENTS FROM GENERATE...",documents)

    # response=retrieval_chain.invoke({"input":"Scaled Dot-Product Attention"})
    # response['answer']
    # generation = retrieval_chain.invoke({"input": question})['answer']
    # generation = retrieval_chain.invoke(question)
    # print("documents...",documents)
    generation = retrieval_chain.invoke({"question":question,"context":documents})




    return {"documents": documents, "question": question, "generation": generation}

  def decide_to_generate(self,state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    # state["question"]
    web_search = state["web_search"]
    # state["documents"]

    if web_search == "Yes":
      # All documents have been filtered check_relevance
      # We will re-generate a new query
      return "transform_query"
    else:
      # We have relevant documents, so generate answer
      return "generate"


  def decide_to_technical(self,state):
    """
    Determines whether the question should be answered by rag pipeline or by the bot.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    # state["question"]
    # technical = state["technical"]
    technical = state["technical"]
    if technical == "technical":
      # All documents have been filtered check_relevance
      # We will re-generate a new query
      return "technical"
    else:
      # We have relevant documents, so generate answer
      return "non-technical"




  def __call__(self,text,retriever):
    self.retriever = retriever
    self.set_tools(text)
    workflow = StateGraph(State_Corrective_Rag) ### StateGraph with AgentState
    workflow.add_node("retrieve",self.retrieve)
    workflow.add_node("grade_documents",self.grade_documents)
    workflow.add_node("transform_query",self.transform_query)
    workflow.add_node("web_search_node",self.web_search)
    workflow.add_node("generate",self.generate)
    workflow.add_node("general_llm",self.general_llm)
    workflow.add_node("question_grader",self.question_grader)

    workflow.add_edge(START,"question_grader")
    workflow.add_conditional_edges("question_grader",
    self.decide_to_technical,
    {
    "technical":"retrieve",
    "non-technical":"general_llm"
})
    workflow.add_edge("retrieve","grade_documents")
    workflow.add_edge("general_llm",END)
    workflow.add_conditional_edges(
        "grade_documents",
        self.decide_to_generate,
        {
            "transform_query":"transform_query",
            "generate":"generate"
        }
    )
    workflow.add_edge("transform_query","web_search_node")
    workflow.add_edge("web_search_node","generate")
    workflow.add_edge("generate",END)

    self.app=workflow.compile()

    return self.app

def call_bot(query):
   llm = LLMSingleton.get_instance()
   return llm.invoke(query)
  
if __name__ == "__main__":
    # query = "Explain attention mechanism"
    # result = call_bot(query)

    # print(result.content)


    # 1)
    loader=PyPDFLoader('full_stack.pdf')
    docs=loader.load()

    # Combine all text into a single variable
    text = "\n".join(doc.page_content for doc in docs)
    # Example usage:


    retriever = store_text_in_vector_db(text)

    mybot=Chatbot()
    workflow=mybot(text,retriever)
    
    inputs = {"question": "Explain full stack web development. Explain the role of full stack developers."}




    store = workflow.invoke(inputs)


    final_answer = store['generation']
    pprint(final_answer)