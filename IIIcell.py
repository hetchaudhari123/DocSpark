from jobspy import scrape_jobs
import pandas as pd
from pprint import pprint
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph import StateGraph,MessagesState, START, END
import PyPDF2
import assemblyai as aai

from typing import List


import os
import sys
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from typing import Annotated, Literal, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.utilities import GoogleSerperAPIWrapper



from langchain_community.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv

import re
import json

import Library as lb

load_dotenv()
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


os.environ["ASSEMBLY_AI_KEY"]=os.getenv('ASSEMBLY_AI_KEY')
aai.settings.api_key = os.getenv("ASSEMBLY_AI_KEY")


os.environ["GROQ_API_KEY"]= os.getenv("groq_api_key")

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')


os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')



def serper_tool(role):
    # Define search queries for each platform
    prompt_coursera = f"Course links for {role} on Coursera"
    prompt_udemy = f"Course links for {role} on Udemy"
    prompt_gfg = f"Course links for {role} on Geeks for Geeks"
    prompt_skillshare = f"Course links for {role} on Skillshare"
    prompt_linkedin_learning = f"Course links for {role} on LinkedIn Learning"

    # Initialize the search API
    search = GoogleSerperAPIWrapper()

    # Define a function to search and return the results for a given query
    def search_platform(prompt):
        response = search.results(prompt)
        results = []
        # Collect valid results (both title and link present)
        for item in response.get('organic', []):
            title = item.get('title', 'nan')
            link = item.get('link', 'nan')
            if title != 'nan' and link != 'nan':  # Only include valid results
                results.append({'title': title, 'link': link})
        return results

    # Get search results for each platform
    coursera_results = search_platform(prompt_coursera)
    udemy_results = search_platform(prompt_udemy)
    skillshare_results = search_platform(prompt_skillshare)
    linkedin_learning_results = search_platform(prompt_linkedin_learning)

    # Initialize an empty list to store the final selected results
    selected_courses = []
    index = 0

    # Sequentially select courses from each platform in a round-robin manner
    while len(selected_courses) < 5:
        # Get course from each platform at the current index
        if index < len(coursera_results):
            coursera_course = coursera_results[index]
            if 'title' in coursera_course and 'link' in coursera_course:
                selected_courses.append(coursera_course)

        if index < len(udemy_results):
            udemy_course = udemy_results[index]
            if 'title' in udemy_course and 'link' in udemy_course:
                selected_courses.append(udemy_course)

        if index < len(skillshare_results):
            skillshare_course = skillshare_results[index]
            if 'title' in skillshare_course and 'link' in skillshare_course:
                selected_courses.append(skillshare_course)
        if index < len(linkedin_learning_results):
            linkedin_learning_course = linkedin_learning_results[index]
            if 'title' in linkedin_learning_course and 'link' in linkedin_learning_course:
                selected_courses.append(linkedin_learning_course)

        # If we have filled 5 results, break out of the loop
        if len(selected_courses) >= 5:
            break
        
        # Increment the index to select the next course from each platform
        index += 1

        if(index == 3):
          break


    # Limit the result to the top 5 available links
    return selected_courses[:5]


def fetch_jobs(
    site_names=["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"],
    search_term="software engineer",
    location="India",
    results_wanted=1,
    hours_old=72,
    country_indeed="India"
):
    """
    Fetches job postings from specified sites based on the given parameters.

    Parameters:
        site_names (list): List of job boards to scrape (e.g., ["indeed", "linkedin"]).
        search_term (str): The term to search for jobs.
        google_search_term (str): Google-specific search query.
        location (str): Job location.
        results_wanted (int): Number of job results desired.
        hours_old (int): Maximum age of job postings in hours.
        country_indeed (str): Country-specific setting for Indeed.
        linkedin_fetch_description (bool): Whether to fetch job descriptions from LinkedIn.
        proxies (list or None): List of proxies for scraping.

    Returns:
        list: A list of job postings.
    """
    google_search_term = f"{search_term} jobs near India since yesterday"
    pd_jobs = scrape_jobs(
        site_name=site_names,
        search_term=search_term,
        google_search_term=google_search_term,
        location=location,
        results_wanted=results_wanted,
        hours_old=hours_old,
        country_indeed=country_indeed,
        verbose=0
    )
    return pd_jobs



def format_jobs_as_text(dataframe):
    """
    Converts a dataframe containing job details into a formatted string,
    listing each job one after another in textual format.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing job details.

    Returns:
    str: A formatted string of job details.
    """
    # Initialize an empty list to store each job's details as a string
    job_texts = []

    # Iterate through the dataframe rows
    for index, row in dataframe.iterrows():
        # Format job details into a block of text
        job_text = (
            f"Site: {row['site']}\n"
            f"Job URL: {row['job_url']}\n"
            f"Title: {row['title']}\n"
            f"Company: {row['company']}\n"
            f"Location: {row['location']}\n"
            f"Date Posted: {row['date_posted']}\n"
            f"Company URL: {row['company_url']}\n"
            "----------------------------------------"
        )
        # Append to the list of job texts
        job_texts.append(job_text)

    # Join all job texts with a newline
    result = "\n".join(job_texts)

    return result

def create_retrieval_grader(llm_with_tool):
    """
    Creates a retrieval grader to query and retrieve relevant courses and their links
    based on a job position and job listings. The grader leverages a system prompt,
    an LLM instance, and a structured output parser for precise results.

    Returns:
        retrieval_grader: A configured grader instance that combines a system prompt,
                          an LLM singleton instance, and a string output parser
                          to produce concise and actionable output.

    Components:
        - System Prompt: Instructs the grader to find courses and their links required
          for excelling at the specified job, while avoiding unnecessary verbose text.
        - LLM Singleton: A singleton instance of the LLM (retrieved via `LLMSingleton.get_instance()`).
        - Output Parser: Ensures the output is formatted as plain text, without additional verbose language.
    """
    system = """Given a job role and some job listings,and message from the tool. Determine the courses
    and their links which are required for excelling the job position.
    You have the access of the following tool:
    search: Does web search.
    If 'messages' has been provided, then that means the tool has provided you a message based on its results.
    Use the tool in case if messages is not there, otherwise output two things: course title, course url.
    Do not add verbose like 'i will search. These are the results,etc.'
    Just output what has been told to output
    Make sure you don't include <\n> in the links and only output links if obtained from the messages of the tool,
    don't make them up.
    """

    # Create the ChatPromptTemplate
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Job Position: \n\n {job_position} \n\n Job Listing: {job}. Messages(if any from the search tool):{messages}"),
        ]
    )

    # Combine the prompt template with the llm_with_tool
    retrieval_grader = grade_prompt | llm_with_tool

    return retrieval_grader




class JobFormat(BaseModel):
  """
    Represents the structure for job and course information. 

    This class is designed to model the structured output from a workflow or a model, making it 
    easier to process and display information related to job opportunities and courses.

    Attributes:
        job_url (List[str]): 
            A list of URLs for job postings.
        job_title (List[str]): 
            A list of job titles corresponding to the provided URLs.
        job_company (List[str]): 
            A list of companies offering the respective jobs.
        job_location (List[str]): 
            A list of locations for the job opportunities.
        job_company_url (List[str]): 
            A list of URLs for the respective companies.
        course_title (List[str]): 
            A list of titles for recommended courses.
        course_link (List[str]): 
            A list of URLs for the respective courses.

    Usage:
        This class is primarily used for handling structured data output for job and course 
        recommendations. It ensures uniformity in accessing and processing the details required for 
        displaying or utilizing the information in applications.
    """

  job_url: List[str] = Field(
      description="A list of job urls."
  )
  job_title: List[str] = Field(
      description="A list of job titles."
  )
  job_company: List[str] = Field(
      description="A list of job companies."
  )
  job_location: List[str] = Field(
      description="A list of job locations."
  )
  job_company_url: List[str] = Field(
      description="A list of job company urls."
  )
  course_title: List[str] = Field(
      description="A list of course titles."
  )
  course_link: List[str] = Field(
      description="A list of course links."
  )


def format_job_courses_as_pandas():
  system = f"""
  You are tasked with extracting and formatting data from job_listings and course_listings.
  Your goal is to produce the following output:
  - job_url: a list of strings representing the URLs of the jobs.
  - job_title: a list of strings representing the titles of the jobs.
  - job_company: a list of strings representing the companies of the jobs.
  - job_location: a list of strings representing the locations of the jobs.
  - job_company_url: a list of strings representing the URLs of the companies.
  - course_title: a list of strings representing the titles of the courses.
  - course_link: a list of strings representing the links to the courses.

  The order of these properties must correspond:
  - For job_url at index 0, it should correspond to job_title at index 0, and similarly for all other job-related properties.
  - The same applies for course properties: course_title at index 0 should correspond to course_link at index 0, and so on.

  Ensure that the properties are directly extracted from the provided inputs without inventing any details.
  Do not change the order of job and course listings, but maintain the internal order for each category.

  You will be given:
  - job_listings
  - course_listings

  Your response should strictly follow this structure.

  """

  llm = lb.LLMSingleton.get_instance()
  llm_with_structured_output = llm.with_structured_output(JobFormat)



  grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Job Listings: \n\n {job_listings} \n\n Course Listings: {course_listings}."),
    ]
  )

  # Combine the prompt template with the llm_with_tool
  retrieval_grader = grade_prompt | llm_with_structured_output

  return retrieval_grader




def extract_json_from_response(response: str) -> dict:
    """
    Extracts a JSON-like structure wrapped in ```json ... ``` from a string response.

    Args:
        response (str): The string response containing the JSON-like structure.

    Returns:
        dict: Parsed JSON data as a Python dictionary.
        None: If no valid JSON is found.
    """
    json_regex = r'```json\n(.*?)\n```'  # Regex pattern to extract JSON block

    # Search for the JSON-like structure
    json_match = re.search(json_regex, response, re.DOTALL)

    if json_match:
        try:
            json_string = json_match.group(1)  # Extract the JSON-like string
            data = json.loads(json_string)  # Parse the JSON string into a Python dictionary
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No valid JSON found.")
        return None

class State(MessagesState):
    """
    Represents the state of job and course information.

    Attributes:
        input (str): The name of the job that the user types in, serving as the query for relevant job information.
        job (str): A textual description containing all the job openings available on the internet for the specified job.
        course (str): A textual description of the recommended courses to excel in the specified job openings.
        table (str): A string representation of a table containing links to job postings and relevant course resources.

    Example:
        - `input` specifies the job the user is searching for, e.g., "Data Scientist".
        - `job` contains detailed descriptions of job openings for "Data Scientist".
        - `course` provides information about courses that enhance skills for "Data Scientist" roles.
        - `table` presents a tabular representation (as a string) of job links and course links for easy access.
        -`final_dict` (dict): A structured dictionary combining job and course data for easy programmatic access.

    """
    job: str
    course: str
    table: str
    input: str
    final_dict: dict

class Chatbot_job_finder:
  def __init__(self):
    self.llm = lb.LLMSingleton.get_instance()

    # self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


  def set_tools(self):
    self.search = TavilySearchResults()
    self.tools = [self.search]
    self.tool_node=ToolNode(self.tools)
    self.llm_with_tools = self.llm.bind_tools(self.tools)
    # self.retrieval_grader = format_job_courses_as_pandas(self.llm)


  def search_jobs(self,state):
    """
    Search for relevant job postings based on the user's input.

    Args:
      state (dict): The current state of the graph, containing the user's input for the job search.
                    Expected to have a key `"input"` that specifies the job title or search term.

    Returns:
      dict: A dictionary containing the search results with job information.
            The result includes a key `"job"` with a string of formatted job descriptions.

    """

    print("<----------search_jobs----------->")

    input = state["input"]

    pd_jobs = fetch_jobs(search_term = input)
    jobs_text = format_jobs_as_text(pd_jobs)

    return {"job":jobs_text}

  def search_courses(self,state):
    """
    Used to split the text into user mentioned number of sub-topics.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates text such that now it contains user mentioned number of topics with
        concepts below each topic.
    """

    print("<----------search_courses----------->")

    job = state["job"]
    input = state["input"]
    messages = state.get("messages",[])

    retrieval_grader = create_retrieval_grader(self.llm_with_tools)
    res = retrieval_grader.invoke({"job_position":input,"job":job,"messages":messages})



    return {"course": res,"messages":res}




  def router_function(self,state):
    #print(f"here is a state from should continue {state}")

    course = state['course']
    #print(f"here is a last message from should continue {last_message}")
    if course.tool_calls:
        print("called tools")
        return "tools"
    return "combiner"

  def combiner(self,state):
    print("<----------combiner----------->")
    job_listings = state["job"]
    course_listings = state["course"]
    # results = self.retrieval_grader.invoke({"job_listings":job_listings,"course_listings":course_listings})
    # parsed_data = extract_json_from_response(results.content)

    # return {"final_dict":parsed_data}
    concept_splitter_llm = format_job_courses_as_pandas()
    response = concept_splitter_llm.invoke({"job_listings":job_listings,"course_listings":course_listings})
    return {"final_dict":response}
  # print(type(concept_splitter_llm))


  # def tool_node(self,state):
  #   results = self.search.invoke("search")
  #   return {"tool_response":results}


  def __call__(self):
    self.set_tools()
    workflow = StateGraph(State) ### StateGraph with AgentState
    workflow.add_node("search_jobs",self.search_jobs)
    workflow.add_node("search_courses",self.search_courses)
    workflow.add_node("tools", ToolNode(self.tools))
    # workflow.add_node("tools", self.tool_node)
    workflow.add_node("combiner", self.combiner)


    workflow.add_edge(START,"search_jobs")
    workflow.add_edge("search_jobs","search_courses")
    workflow.add_edge("tools","search_courses")
    workflow.add_conditional_edges(
        "search_courses",
        self.router_function,
         {
             "tools": "tools",
             "combiner": "combiner"
        })
    workflow.add_edge("combiner",END)


    self.app=workflow.compile()

    return self.app
  

if __name__ == "__main__":
    # mybot=Chatbot_job_finder()
    # workflow=mybot()
    # response=workflow.invoke({"input": "Software Engineer"})
    # print(response)
    role = "Software Engineering"
    top_courses = serper_tool(role)

    # Output the results
    for course in top_courses:
        print(f"Title: {course['title']}\nLink: {course['link']}\n")