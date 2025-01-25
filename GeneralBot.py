from langgraph.graph import StateGraph,MessagesState, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from pprint import pprint
import Library as lb
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import tools_condition # this is the checker for the
from IPython.display import Image, display




import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"]= GROQ_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]=LANGCHAIN_PROJECT




    













class General_Bot:
    def __init__(self):
        self.llm = lb.LLMSingleton.get_instance()




    def set_tools(self):
        search = TavilySearchResults()
        self.tools = [search]
        self.tool_node=ToolNode(self.tools)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    

    def reasoner(self,state:MessagesState):
        """Reasoner method deciding which tool to be used or generate the answer."""
        msg=state["messages"]
        sys_msg=SystemMessage("""You are a helpful assistant having access to the tool:
                            1)search:for searching the web.
                              Please output according to the prompt without mentioning that you have used the tool even if you did.""")
        # sys_msg = SystemMessage(content="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs.")
        result=self.llm_with_tools.invoke([sys_msg]+msg)
        return {"messages":[result]}


 






    def __call__(self):
        self.set_tools()
        workflow=StateGraph(MessagesState)
        workflow.add_node("reasoner",self.reasoner)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_edge(START,"reasoner")
        workflow.add_conditional_edges(
        "reasoner",
        tools_condition)
        workflow.add_edge("tools", "reasoner")

        app=workflow.compile()
        return app
  
if __name__ == "__main__":

    # Instantiate the bot
    gbot = General_Bot()
    app = gbot()

    # Invoke the app with a weather query
    # results = app.invoke({"messages": "What is the weather of New York?"})
    results = app.invoke({"messages": "Explain attention mechanism"})
    
    # Print the assistant's response
    print(results['messages'][-1].content)
