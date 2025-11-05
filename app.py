import os
import assemblyai as aai

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

