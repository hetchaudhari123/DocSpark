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


GROQ_API_KEY=os.getenv('groq_api_key')
os.environ['GROQ_API_KEY']=GROQ_API_KEY


os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]= "Rag Tool for NLP"

os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')

os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

