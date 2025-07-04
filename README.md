# DocSpark

![DocSpark Homepage](/assets/homepage.png)

DocSpark is an intelligent document processing and interaction application that leverages advanced Natural Language Processing (NLP) models like LangGraph and corrective RAG (retrieval-augmented generation). It allows users to upload documents, take quizzes, get job-related information, and more. The app is designed to assist in learning, job exploration, and general inquiries.
## Features
### Library Section
- Users can upload documents (PDF or Video) and ask questions related to the content.
- The Library Bot utilizes corrective RAG to provide answers based on the uploaded documents.
### Teacher Section
- Users can request a 50-mark question paper with 5 questions (each worth 10 marks).
- After uploading their answers, the system automatically grades the responses and generates a detailed report indicating the total score.
### III Cell Section
- Users can enter the name of a job they aim for.
- The system performs a web search to show open positions for that job in India.
- Additionally, it suggests courses that could help the user excel in the desired job.
### General Bot
- Users can ask any question, and the General Bot provides responses while maintaining past conversations for context.
## Installation
To run DocSpark locally, follow these steps:
1. Clone the repository:
    ```bash
    https://github.com/hetchaudhari123/DocSpark.git
    ```
2. Navigate to the project directory:
    ```bash
    cd DocSpark
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Start the application:
    ```bash
    streamlit run streamlit_app.py
    ```
## Technologies Used
- **LangGraph** for defining the various workflows of the application's features.
- **Corrective RAG** for retrieval-augmented generation-based answers from documents.
- **Streamlit** for building the web application and creating the interactive user interface.
- **JobSpy** for web scraping job data in the III Cell Section and searching for job openings.
- **Llama-3.1-70B-Versatile Model** for advanced language model support in document processing and bot interactions, used in the Library, III Cell, and General Bot sections.
- **Gemini-1.5-Flash Model** for enhanced natural language understanding and generation specifically in the Teacher section, for question paper generation, automatic grading,answer sheet evaluation and report generation.
## Usage
- **Library Section:**
  - Upload your documents (PDF or Video) and interact with the bot to get answers related to the content.
- **Teacher Section:**
  - Request a custom question paper and upload your answers to receive detailed grading feedback.
- **III Cell Section:**
  - Enter a job title, and the system will provide current open job positions in India along with suggested courses to pursue.
- **General Bot:**
  - Ask any question, and the bot will reply while maintaining conversation context.
## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
## Acknowledgements
- **LangChain** and **LangGraph** for advanced NLP capabilities.
- **Corrective RAG** for improving document interaction and answer generation.
- **Llama and Gemini Models** for enhancing chatbot functionality.
