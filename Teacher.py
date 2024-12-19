import os
from typing import Annotated, Literal, TypedDict
import os
import PyPDF2
import assemblyai as aai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader


from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

import random



from langchain.chains import create_retrieval_chain

from langchain_core.runnables import RunnablePassthrough

import re
import json

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



from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import langchain_core
from typing import List
import langchain_community

from pprint import pprint

from langchain_community.tools.tavily_search import TavilySearchResults

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import Library as lb

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





class SubtopicFormat(BaseModel):
    """
    Represents the structure of a document containing various topics and their associated contexts.

    This class is designed to model the structure of a text where each topic is followed by its
    corresponding textual description (context). It is primarily intended for use cases where text
    needs to be divided into subtopics.

    Attributes:
        topics (List[str]):
            A list of topics identified from the text. Each topic acts as a header or a subheading
            summarizing the content it covers.
        context (List[str]):
            A list of textual descriptions corresponding to each topic. These descriptions provide
            detailed information or explanations about the respective topics.

    Example Structure:
        - `topics`: A list of identified topics.
        - `context`: A list of text sections providing the content for each topic.

    Usage:
        This class can be used to model outputs from a text processing system, LLMs (Large Language Models),
        or other tools that need to split and organize text into topics and contexts. It ensures that the
        text is structured for readability and usability in question generation.
    """

    topics: List[str] = Field(
        description="A list of topics identified from the text."
    )

    context: List[str] = Field(
        description="A list of textual descriptions representing the detailed information for each topic."
    )

def llm_concept_splitter():
    """
    Function to create and return the concept splitter LLM.

    The LLM is designed to split a given context into a user-specified number of distinct sub-topics.
    Each sub-topic will be presented as a title in the `topics` list, followed by the relevant
    portion of the text corresponding to that sub-topic in the `context` list. The aim is to
    ensure that the entire text is effectively divided into cohesive sections, maintaining
    a one-to-one mapping between topics and their respective contexts.

    Output Structure:
        - `topics`: A list of topic titles either extracted or created from the input context.
        - `context`: A list of textual descriptions, where each description corresponds to the sub-topic
          at the same index in the `topics` list. For example, the context at `index=0` will describe
          the topic at `index=0`, and so on.
        - One-to-one Mapping: The order of topics and contexts is preserved, ensuring consistency
          and ease of reference.

    Notes:
        - The generated sub-topics should not include meta-text, such as "This section deals with XYZ subtopic."
        - If the provided context does not have sufficient content to generate the requested number of sub-topics,
          the LLM will generate as many sub-topics as possible.
        - This function uses a structured output format to ensure the output adheres to the specified structure.

    Returns:
        A callable object capable of splitting the input text into sub-topics and organizing the content in a
        structured format with one-to-one mapping between topics and contexts.
    """
    # llm = lb.LLMSingleton.get_instance()
    llm = lb.GeminiSingleton.get_instance()

    # Define the system prompt for rewriting the question
    system = f"""You will be given a text and tasked with dividing it into distinct topics.
    For each topic:
    1. Extract the title and include it in a list of topics.
    2. Extract the relevant portion of the text for that topic and include it in a corresponding list of contexts.

    Guidelines:
    - Ensure that each topic includes only relevant concepts or ideas from the context. Avoid adding meta-text like "This section deals with XYZ."
    - Maintain a one-to-one mapping between topics and contexts, such that the context at index 0 corresponds to the topic at index 0, and so on.
    - If the provided context does not contain enough material to generate the specified number of topics, create as many meaningful topics as possible without repeating or fabricating content.
    - Do not include unnecessary topics like 'course evaluation', 'name of the research paper' rather include conceptual topics.
    Your output should strictly be json string containing exactly two keys namely topic and context. Do not add anything else(like metadata as json:) apart from this structure.
    For example if text related to Full Stack Web Development is provided, then it can create topics(based on the text) like:
    "topic":["Full Stack Development Basics","Full Stack Developers and Their Role"],
    "context":["Full stack development involves '
 'designing, creating, testing, and deploying a complete web application. It '
 'encompasses front-end, back-end, and database development. Full-stack '
 'developers possess expertise in both front-end and back-end technologies, '
 'handling the entire web application development process. They work with '
 'technologies like HTML, CSS, JavaScript, PHP, Ruby on Rails, and Node.js.  '
 'Full stack development includes both front-end (user interface) and back-end '
 '(business logic and workflows).  A retail website example demonstrates the '
 'interplay of front-end technologies (HTML, CSS, Javascript), back-end '
 'languages (Java, Python) with frameworks like SpringBoot or Django, and '
 'databases for storing user and transaction data.\n'","Full stack developers must be '
 'proficient in an entire technology stack.  For instance, MEAN stack '
 'developers work with MongoDB, Express, Angular, and Node.js.  They choose '
 'appropriate technologies, write clean code, and stay updated with the latest '
 'tools.  Responsibilities include selecting technologies, writing clean code, '
 'and keeping up with advancements.  They frequently use JavaScript for both '
 'front-end and back-end development. Other popular languages include HTML, '
 'CSS, Python, Java, R, Ruby, Node.js, and PHP. Technology stacks like MEAN, '
 'MERN, Ruby on Rails, and LAMP are commonly used."]
 """

    # Define the rewrite prompt template
    concept_split_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the text:{text}."
            ),
        ]
    )

    llm_with_structured_output = llm.with_structured_output(SubtopicFormat)

    # Create the question rewriter with the defined prompt and output parser
    concept_splitter = concept_split_prompt | llm | StrOutputParser()
    # concept_splitter = concept_split_prompt | llm_with_structured_output



    # Return the question rewriter
    return concept_splitter

def extract_llm_concept_splitter():
    """
    Function to create and return the concept splitter LLM.

    The LLM is designed to split a given context into a user-specified number of distinct sub-topics.
    Each sub-topic will be presented as a title in the `topics` list, followed by the relevant
    portion of the text corresponding to that sub-topic in the `context` list. The aim is to
    ensure that the entire text is effectively divided into cohesive sections, maintaining
    a one-to-one mapping between topics and their respective contexts.

    Output Structure:
        - `topics`: A list of topic titles either extracted or created from the input context.
        - `context`: A list of textual descriptions, where each description corresponds to the sub-topic
          at the same index in the `topics` list. For example, the context at `index=0` will describe
          the topic at `index=0`, and so on.
        - One-to-one Mapping: The order of topics and contexts is preserved, ensuring consistency
          and ease of reference.

    Notes:
        - The generated sub-topics should not include meta-text, such as "This section deals with XYZ subtopic."
        - If the provided context does not have sufficient content to generate the requested number of sub-topics,
          the LLM will generate as many sub-topics as possible.
        - This function uses a structured output format to ensure the output adheres to the specified structure.

    Returns:
        A callable object capable of splitting the input text into sub-topics and organizing the content in a
        structured format with one-to-one mapping between topics and contexts.
    """
    # llm = lb.LLMSingleton.get_instance()
    llm = lb.GeminiSingleton.get_instance()

    # Define the system prompt for rewriting the question
    system = f"""You are given a text string containing two fields keys and context. Your task is to convert it into json
    containing those two fields. Do not add anything extra just create the json from the string. Make sure that the length
    of the concept list and topics list is the same.
 """

    # Define the rewrite prompt template
    concept_split_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the text:{text}."
            ),
        ]
    )

    llm_with_structured_output = llm.with_structured_output(SubtopicFormat)

    # Create the question rewriter with the defined prompt and output parser
    concept_splitter = concept_split_prompt | llm_with_structured_output



    # Return the question rewriter
    return concept_splitter


def select_random_unique_contexts(contexts, num_contexts):
    """
    Selects a specified number of unique random contexts from the given list.

    Args:
        contexts (list): A list of contexts.
        num_contexts (int): The number of random contexts to select.

    Returns:
        list: A list containing the randomly selected unique contexts.
    """
    if num_contexts > len(contexts):
        return contexts

    # Randomly select unique contexts
    selected_contexts = random.sample(contexts, num_contexts)

    return selected_contexts



class QuestionFormat(BaseModel):
  """
  Represents the structure of a question paper with questions, corresponding marks, and associated concepts.

  This class is used for modeling questions, marks, and the topics related to each question. It is
  designed to structure the output from an LLM (Large Language Model) so that it can be easily parsed
  and used for further processing, like generating question papers or quizzes.

  Attributes:
    questions (List[str]): A list of questions that need to be asked.

  Each attribute ensures that the output from the LLM will be structured as:
  - `questions`: The questions the model generates.
  """

  questions: List[str] = Field(
      description="A list of questions present in the question paper."
  )



class State(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        text (str): The textual data regarding the topic.
        question (List[str]): A list of questions.
        mark (List[int]): A list of integers representing the marks for each corresponding question.
        topic (List[str]): A list of topics, where each index corresponds to the topic for the question at the same index.

    Example:
        - question[0] corresponds to mark[0], topic[0], and score[0].
        - For each question, the `topic`, `mark`, and `score` at the same index are related.
    """
    text: str
    question: List[str]
    mark: List[int]
    topic: List[str]
    input:str
    # theory:str

class Chatbot_Question_Generator:
  def __init__(self,retriever,questions):
    self.retriever = retriever
    self.questions = questions
    self.llm = lb.GeminiSingleton.get_instance()
    self.llm_concept_splitter = llm_concept_splitter()
    self.web_search_tool = TavilySearchResults(k=3)
    self.llm_extract_splitter = extract_llm_concept_splitter()







  def concept_splitter(self,state):
    """
    Used to split the text into user mentioned number of sub-topics.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates text such that now it contains user mentioned number of topics with
        concepts below each topic.
    """

    input = state["input"]




    concept_split_text = self.llm_concept_splitter.invoke({"text":input})
    response = self.llm_extract_splitter.invoke({"text":concept_split_text})
    print("length of topics",len(response.topics))
    print("length of context",len(response.context))
    # result = select_random_unique_topics(response.topics, response.context, num_topics = self.questions)
    result = select_random_unique_contexts(response.context, self.questions)
    print("The length of the selected context...",len(result))

    return {"text": result}

  def test_generator(self,state):
    """
    Generates questions depending upon the topics identified.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Creates question key containing the questions generated, topic key containing topics list, mark key containing marks_list
    """

    llama = self.llm
    text = state["text"]





    model_with_structured_output=llama.with_structured_output(QuestionFormat)
    question_paper = model_with_structured_output.invoke(f"""Given the list of contexts: {text}.
    Create 10 marker descriptive question that can be asked in a question paper corresponding to each context mentioned.
    Please don't add anything else apart from the questions.
    Don't make more than one question for one context
    Make sure you only make {self.questions} number of questions.""")
    questions_list = question_paper.questions
    # topics_list = question_paper.topics
    marks_list = []
    for i in questions_list:
      marks_list.append(10)


    # return {"question":questions_list,"topic":topics_list,"mark":marks_list}
    return {"question":questions_list,"mark":marks_list}



  def __call__(self):
    # self.set_tools(text)
    workflow = StateGraph(State) ### StateGraph with AgentState
    # workflow.add_node("search_from_db",self.search_from_db)
    # workflow.add_node("search_from_net",self.search_from_net)
    workflow.add_node("concept_splitter",self.concept_splitter)
    workflow.add_node("test_generator",self.test_generator)

    workflow.add_edge(START,"concept_splitter")
    workflow.add_edge("concept_splitter","test_generator")
    workflow.add_edge("test_generator",END)

    self.app=workflow.compile()

    return self.app


# print(response['messages'][-1].content)

class Report_Format(BaseModel):
    """
    Represents the structured format of the report for a question paper attempted by the user.

    Attributes:
        score (List[int]): A list of marks obtained by the user for each question, corresponding to the order of the questions.
        marks (List[int]): A list of total marks allocated for each question in the paper.
        question (List[str]): A list of all the questions included in the paper, presented in sequential order.
        reason (List[str]): A list of reasons explaining the scores awarded for each question, providing insights into user performance.
    """
    score: List[int] = Field(
        description="Marks obtained by the user for each question, in the same order as the questions."
    )




def llm_extract_components():
    # Define the system prompt
    system = """Given a text formatted in the way such that it contains: Question, Answer, Score, Reason, then next again: Question, Answer, Score, Reason and so on.
    Return a score list extracting the score present for each (Question, Answer, Score, Reason).
    E.g.
    For the input:
    1)Question:"question1"
    Answer:"ans1"
    Score:"<Score1>/<mark1>"
    Reason:"reason1"

    2)Question:"question2"
    Answer:"ans2"
    Score:"<Score2>/<mark2>"
    Reason:"reason2"

    3)Question:"question3"
    Answer:"ans3"
    Score:<Score3>/<mark3>
    Reason:"reason3"

    Output Expected:
    score:[Score1, Score2, Score3]
    Please Make sure that the score is list of integers otherwise error will occur. So make sure score is a list of integers.
    Make sure that you extract the score properly without missing any score, there would be in total 5 scores in the text.
    """

    # Initialize the LLM and set the output structure
    llm = lb.GeminiSingleton.get_instance()
    # llm = CohereSingleton.get_instance()

    # Create the prompt template
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """Here is the text:{text}
                """,
            ),
        ]
    )

    # Chain the grader and invoke the model
    structured_llm_grader = llm.with_structured_output(Report_Format)

    grader = re_write_prompt | structured_llm_grader

    return grader

def generate_report(question_list, answer_list, score_list, mark_list, reason_list, topics_list):
    """
    Generates a formatted report based on the provided question paper data.

    Parameters:
        question_list (List[str]): List of all questions in the paper.
        answer_list (List[str]): List of answers provided for each question.
        score_list (List[int]): Marks obtained for each question.
        mark_list (List[int]): Total marks allocated for each question.
        reason_list (List[str]): Explanation for the awarded marks for each question.
        topics_list (List[str]): List of topics corresponding to each question.

    Returns:
        str: A formatted string report.
    """
    # Calculate total marks
    total_obtained = sum(score_list)
    total_marks = sum(mark_list)

    # Start building the report
    report = []
    report.append(f"Total Marks Obtained: {total_obtained}/{total_marks}\n")
    report.append("Details:\n")

    # Track topics strong and weak at
    topic_performance = {}

    # Add details for each question
    for i in range(len(question_list)):
        report.append(f"Question {i + 1}: {question_list[i]}")
        report.append(f"Answer Written: {answer_list[i]}")
        report.append(f"Score Awarded: {score_list[i]}/{mark_list[i]}")
        report.append(f"Reason: {reason_list[i]}\n")

        # Calculate percentage marks for the question
        percentage = (score_list[i] / mark_list[i]) * 100

        # Update topic performance
        topic = topics_list[i]
        if topic not in topic_performance:
            topic_performance[topic] = []
        topic_performance[topic].append(percentage)

    # Determine strong and weak topics based on thresholds
    topics_strong_at = []
    topics_weak_at = []

    for topic, percentages in topic_performance.items():
        average_score = sum(percentages) / len(percentages)
        if average_score >= 70:
            topics_strong_at.append(topic)
        elif average_score <= 30:
            topics_weak_at.append(topic)

    # Add topics strong and weak at to the report
    report.append("Topics Strong At:")
    report.append(", ".join(topics_strong_at) if topics_strong_at else "None")
    report.append("\n")
    report.append("Topics Weak At:")
    report.append(", ".join(topics_weak_at) if topics_weak_at else "None")

    # Return formatted report as a single string
    return "\n".join(report)



def evaluate_answers(question_sheet, answer_sheet, marks, theory):
    # Define the system prompt
    system = """Given a list of questions, answers, corresponding marks for each question, and the complete theory from which the questions were derived, your task is to evaluate the answers based on their quality and correctness relative to the provided theory.

    For each question, assess the answer by comparing it with the relevant theory to determine its accuracy, completeness, and alignment with the expected concepts. Assign marks based on the grading criteria provided in the marks list, ensuring that the total marks for each question are adhered to.

    After evaluating all answers, return a list of scores, where each score represents the marks obtained by the student for each question. Ensure that the scores correspond to the questions, answers, and marks in the same order as they were provided.
    Also keep in mind the quality of the answer while evaluating. Some answers might be partial or not at all relevant to the question, grade them accordingly.
    Don't give some code for doing evaluation, you yourself evaluate.

    E.g.
    1)Question:"Give four example of proper nouns."\\n
    Answer:"Sun,moon".\\n
    Score:"<half_of_the_total_marks_for_that_question>/<total_marks_for_that_question>"\\n
    Reason:"Four nouns are asked but user has answered only two, thus half question is answered correctly, thus 5 out of 10."\\n

    2)Question:"Explain attention mechanism."\\n
    Answer:"The sun is a star."\\n
    Score:"<0>/<total_marks_for_that_question>"\\n
    Reason:"User is asked about attention mechanism but answering something else entirely, thus zero marks."\\n

    3)Question:"Explain in detail transformer architecture."\\n
    Answer:"The Transformer architecture is a deep learning model primarily used for natural language processing tasks. It relies on self-attention mechanisms to process input sequences in parallel, unlike traditional models like RNNs that process data sequentially. The model consists of an encoder-decoder structure, where the encoder processes the input data, and the decoder generates the output. The self-attention mechanism allows the model to weigh the importance of different words in a sentence, regardless of their position. Transformers have achieved state-of-the-art performance in tasks like translation, text generation, and sentiment analysis."\\n
    Score:"<3>/<total_marks_for_that_question>"\\n
    Reason:"The answer is correct but is vague and not detailed, thus 3 marks out of 10."\\n
    
    Do not add verbose from your end like 'based on the context, i will generate the report' or 'Here are the evaluations of the answers based on the provided theory', etc.
    
    Just output question, answer,score,reason... as mentioned in the examples.
    Make sure that the question, answer, score , reason should be separated by \\n(backslash n, an escape character) but don't remove the terms Question, Answer, Score, Reason.
    """

    # Initialize the LLM and set the output structure
    llm = lb.GeminiSingleton.get_instance()
    # llm = CohereSingleton.get_instance()
    # structured_llm_grader = llm.with_structured_output(Score)

    # Create the prompt template
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """Here is the question list:{question_sheet}.
                Here is the answer sheet:{answer_sheet}.
                Here is the marks list:{marks}.
                Here is the theory:{theory}
                """,
            ),
        ]
    )

    # Chain the grader and invoke the model
    grader = re_write_prompt | llm | StrOutputParser()

    # Invoke the grader with the inputs
    graded_answers = grader.invoke({
        "question_sheet": question_sheet,
        "answer_sheet": answer_sheet,
        "marks": marks,
        "theory": theory
    })

    # Return the final evaluated answer sheet
    return graded_answers

class State_Report_Generator(TypedDict):
  """
  Represents the state of the graph and provides details for generating a report.

  Attributes:
  answer_sheet (List[str]): A list of answers corresponding to the questions.
  question (List[str]): A list of questions for the report.
  mark (List[int]): A list of integers representing the marks for each corresponding question.
  topic (List[str]): A list of topics where each index corresponds to the topic for the question at the same index.
  topic_good_at (List[str]): A list of topics in which the user has performed well.
  topic_weak_at (List[str]): A list of topics in which the user has shown weakness.
  theory: A string containing the theory from which the question paper was made.

  Example:
  - `question[0]` corresponds to `mark[0]`, `topic[0]`, and `answer_sheet[0]`.
  - `topic_good_at` and `topic_weak_at` provide an overview of strong and weak areas respectively.

  Use Case:
  This class can be used to structure data for evaluating user performance and generating insights about strengths and weaknesses.
  """
  answer_sheet: List[str]
  question: List[str]
  mark: List[int]
  # topic: List[str]
  theory:str
  evaluated_sheet:str
  report:str

class Chatbot_Report_Generator:
  def __init__(self,retriever):
    self.llm = lb.LLMSingleton.get_instance()
    self.retriever = retriever
    self.gemini = lb.GeminiSingleton.get_instance()
    self.llm_concept_splitter = llm_concept_splitter()
    self.web_search_tool = TavilySearchResults(k=3)




  def evaluator(self,state):
    """
    Evaluates the answer sheet.

    Args:
        state (dict): The current graph state containing the user's input or question.

    Returns:
        dict: evaluated_sheet containing the scores and the reason for the marks given.


    """


    evaluated_sheet = evaluate_answers(state["question"], state["answer_sheet"], state["mark"],state["theory"])

    return {"evaluated_sheet":evaluated_sheet}

  def report_generator(self,state):
    """
    Generates a detailed report based on the evaluated answers and topics provided in the state.

    Args:
    state (dict): The current graph state, expected to include:
      - `evaluated_sheet` (str): A string containing the evaluated answers in a structured format.
      - `topic` (List[str]): A list of topics corresponding to each question.

    Returns:
    dict: A dictionary containing the generated report with the following keys:
      - `report` (str): A detailed textual report including:
          - Total marks obtained and total marks of the paper.
          - A per-question breakdown with:
              - Question text.
              - Answer provided.
              - Score obtained and total marks for the question.
              - Reason for the awarded marks.
          - A summary of topics the user is strong and weak in, determined dynamically based on score thresholds.
    """

    evaluated_sheet = state["evaluated_sheet"]

    extractor = llm_extract_components()
    graded_scores = extractor.invoke({"text":evaluated_sheet})

    total_score = sum(graded_scores.score)

    exam_score = sum(state["mark"])


    # gen_report = generate_report(graded_answers.question,graded_answers.answer,graded_answers.score,graded_answers.mark,graded_answers.reason,state["topic"])

    gen_report = f"{evaluated_sheet}\nTotal Score:{total_score}/{exam_score}"

    return {"report": gen_report}



  def __call__(self):
    workflow = StateGraph(State_Report_Generator) ### StateGraph with AgentState
    workflow.add_node("evaluator",self.evaluator)
    workflow.add_node("report_generator",self.report_generator)

    workflow.add_edge(START,"evaluator")
    workflow.add_edge("evaluator","report_generator")
    workflow.add_edge("report_generator",END)

    self.app=workflow.compile()

    return self.app



def report_generator_using_llm(retriever,question_list,answer_list,mark_list,theory):
    mybot=Chatbot_Report_Generator(retriever)
    workflow=mybot()
    response=workflow.invoke({"question":question_list,"answer_sheet":answer_list,"mark":mark_list,
                              "theory":theory})
    return response["report"]


def question_paper_generator(retriever,text,questions=5):
  mybot=Chatbot_Question_Generator(retriever,questions=5)
  workflow=mybot()
  response=workflow.invoke({"input": text})
  # return response["question"],response["topic"],response["mark"],response["theory"]
  return response['question'],response['mark']


if __name__ == "__main__":
    loader=PyPDFLoader('full_stack.pdf')
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    documents=text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    db = Chroma.from_documents(documents,embeddings)
    retriever=db.as_retriever()
    full_stack_text = "\n".join(doc.page_content for doc in docs)

    # questions_list = question_paper_generator(retriever,full_stack_text,5)
    # pprint(questions_list)

    
    answer_sheet = ["""Full stack development involves several key stages that contribute to building a complete web application.
    It begins with requirement analysis to define project goals and technical specifications, followed by front-end development to design user interfaces and experiences using HTML, CSS, and JavaScript frameworks like React or Angular.
    Back-end development focuses on server-side logic, APIs, and database integration using technologies such as Node.js, Python (Django), or Java (SpringBoot).
    Efficient data storage and retrieval are managed through databases like MongoDB or MySQL.
    Testing and debugging ensure the application's functionality and reliability, while deployment and maintenance bring the application to live environments and handle updates.
    A full stack developer is responsible for overseeing the entire process, requiring skills in front-end and back-end technologies, database management, and version control tools like Git.
    Their expertise ensures seamless integration of all components, delivering a functional and user-friendly application.""",
                    """A full stack developer is responsible for managing both the front-end and back-end development of web applications.
                    Their role includes designing user interfaces, developing server-side logic, creating and integrating APIs, and managing databases.
                    They ensure the seamless operation of all components, troubleshoot issues, and optimize performance.
                    To excel in this role, a full stack developer needs proficiency in front-end technologies like HTML, CSS, and JavaScript, back-end frameworks like Node.js, Django, or SpringBoot, and database systems such as MongoDB or MySQL.
                    Familiarity with version control tools like Git, cloud deployment platforms, and responsive design principles is also essential.
                    Strong problem-solving skills, adaptability, and staying updated with the latest technologies are crucial for success in this dynamic role.""",
    """The 3-tier application architecture divides an application into three logical layers: the presentation layer, business logic layer, and data access layer.
    The presentation layer handles user interactions and displays the interface, the business logic layer processes data and executes the core functionality, and the data access layer interacts with the database to retrieve or store information.
    By separating concerns, this architecture enhances flexibility as changes in one layer, such as switching a database engine or updating the user interface, do not impact others.
    It improves maintainability by isolating code into manageable sections, allowing teams to work independently on different layers.
    Furthermore, reusability is supported as components in one layer, like business rules or database queries, can be reused across multiple applications.
    This modular design makes the system scalable and easier to test, debug, and upgrade.""",
                    """RESTful APIs are based on several key constraints that ensure effective client-server communication.
                    These include client-server separation, which divides responsibilities between the user interface and data storage for scalability; statelessness, where each request contains all the necessary information, making interactions independent and simplifying server design; and cacheability, allowing responses to be stored and reused to enhance performance.
                    A uniform interface standardizes communication, ensuring consistency through resources identified by URIs and operations defined using HTTP methods like GET, POST, PUT, and DELETE. Additionally, REST supports layered architecture, enabling intermediaries like load balancers to enhance scalability and security.
                    These constraints promote modularity, improve performance, and simplify integration, making RESTful APIs highly effective for web-based systems.""",
    """JSON (JavaScript Object Notation) offers several advantages as a data interchange format in web applications.
    It is lightweight and human-readable, making it easy to understand and debug. JSON is language-independent but integrates seamlessly with most programming languages, enhancing versatility.
    It supports essential data structures like objects (dictionaries) and arrays (lists), representing complex data in a structured and organized way.
    JSON's key-value pair format is intuitive for both developers and systems. Additionally, it facilitates efficient serialization and transmission of data between a server and client, commonly in APIs.
    Its compact syntax, compared to XML, ensures faster parsing and reduced data size, improving application performance."""]


    question_sheet = ['What are the key stages involved in full stack development and how do they '
    'contribute to the creation of a complete web application?',
    'What are the responsibilities of a full stack developer and what skills are '
    'required to be proficient in the role?',
    'Explain the concept of 3-tier application architecture and how it enhances '
    'flexibility, maintainability, and reusability in software development.',
    'Describe the key constraints of RESTful APIs and how they enable effective '
    'communication between client and server.',
    'What are the advantages of using JSON as a data interchange format in web '
   'applications and how does it support data structures and serialization?']
    
    reason=["The answer is extremely vague and doesn't mention any of the stages or how they contribute to creating a web application.  It only reiterates the question.", 
            'This answer is fairly comprehensive, covering key responsibilities and required skills. It could be improved by mentioning testing and specific examples of tools used for different tasks.', 
            'This is a well-structured and accurate explanation of 3-tier architecture, clearly outlining the layers and their benefits regarding flexibility, maintainability, and reusability.',
            'This answer accurately describes the key constraints of RESTful APIs and their benefits.  It would benefit from a more in-depth explanation of HATEOAS, a core constraint of REST.', 
            "This answer provides a thorough explanation of JSON's advantages, including its lightweight nature, readability, language independence, support for data structures, and efficient serialization.  It also correctly positions JSON against alternatives like XML."]

    mark_list = [10,10,10,10,10]


    fake_answer_sheet = answer_sheet
    fake_answer_sheet[0] = "Full stack development involves several key stages that contribute to building a complete web application. "
    # fake_answer = evaluate_answers(question_sheet, fake_answer_sheet, marks, full_stack_text)
    # pprint(fake_answer)
    result = report_generator_using_llm(retriever,question_sheet,fake_answer_sheet,mark_list,full_stack_text)
    print(result)

#     score=[2,8,8,9,7]
    # gen_report = generate_report(question_sheet,fake_answer_sheet,score,marks,reason,topic)
#     pprint(gen_report)
    # extractor = llm_extract_components()
    # graded_answers = extractor.invoke({"text":gen_report})
    # print(graded_answers)
    # results = report_generator_using_llm(retriever,5,question_sheet,answer_sheet,marks,topic,full_stack_text)
    # print(results)