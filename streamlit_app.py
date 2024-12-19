import streamlit as st
import os
import Library as lb
import Teacher as tb
import IIIcell as iii
import pandas as pd
import GeneralBot as gb

# Ensure Chroma client and other components are initialized only once
if "mybot" not in st.session_state:
    st.session_state.mybot = lb.Chatbot()

if "iii_bot" not in st.session_state:
    st.session_state.iii_bot = iii.Chatbot_job_finder()
    
if "iii_workflow" not in st.session_state:
    st.session_state.iii_workflow = st.session_state.iii_bot()







if "gen_bot" not in st.session_state:
    gbot = gb.General_Bot()
    st.session_state.gen_bot = gbot()





# Function to simulate file upload by loading a file from the specified path
def auto_upload_file(file_path):
    """
    Simulate file upload by loading a file from the specified path.
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            file_data = f.read()
            # Simulating file upload by returning the file data
            return file_data
    else:
        st.error("File not found!")
        return None

# Handle file upload logic
import tempfile  # For creating temporary files

def handle_file_upload(uploaded_files):
    all_text = ""  # Initialize a string to accumulate text content


    if uploaded_files:
        # Iterate through each uploaded file
        for file in uploaded_files:
            if hasattr(file, 'name'):
                file_extension = file.name.split('.')[-1].lower()

                # Determine the file type based on the extension
                if file_extension in ["pdf"]:
                    file_type = "PDF"
                else:
                    file_type = "Video"

                # st.write(f"Processing {file_type} file: {file.name}")

                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
                    temp_file.write(file.getvalue())
                    temp_file_path = temp_file.name  # Get the path of the temporary file

                # Call appropriate extraction logic
                try:
                    extracted_text = lb.text_extractor(temp_file_path,file_type)  # Extract text from PDF
                    all_text += extracted_text  # Accumulate extracted content
                except Exception as e:
                    st.error(f"Error processing file {file.name}: {e}")
                finally:
                    # for printing that the file has been added
                    os.remove(temp_file_path)  # Clean up the temporary file
            else:
                st.sidebar.write("File does not have a valid name attribute.")
        
        st.session_state.text = all_text
        print("begin...",st.session_state.text[:100])
        print("end...",st.session_state.text[-100:])
        # Send extracted text to the bot workflow
        st.session_state.retriever = lb.store_text_in_vector_db(all_text)
        workflow = st.session_state.mybot(all_text,st.session_state.retriever)
        st.session_state.workflow = workflow
        st.session_state.uploaded_files = uploaded_files
        return all_text
    else:
        st.sidebar.write("No files uploaded yet!")
        return None


def library_main():
    st.write("## Workflow")

    # if "workflow" in st.session_state:
    #     st.write(st.session_state.workflow)

    def process_input():
        user_input = st.session_state.user_input  # Retrieve input from session state
        if user_input:  # Only process if the input is not empty
            if "workflow" in st.session_state:
                # Append user message to session state messages **only if workflow is ready**
                st.session_state.messages.append({"role": "User", "content": user_input})
                
                # Process input with the bot and get response
                bot_response = st.session_state.workflow.invoke({"question": user_input})['generation']
                st.session_state.messages.append({"role": "Bot", "content": bot_response})
                
                # Clear any error messages
                st.session_state.error_message = ""
            else:
                # Set an error message if the bot workflow is not ready
                st.session_state.error_message = "Please provide the material, and then press 'Add Files'!"
            
            # Clear the input bar
            st.session_state.user_input = ""

    st.write("### Library Section")
    st.subheader("Have questions about your PDFs or videos? Ask them here!")


    # Initialize messages and error state in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if "error_message" not in st.session_state:
        st.session_state.error_message = ""

    # Display conversation history (for chat-like experience)
    for msg in st.session_state.messages:
        if msg['role'] == "User":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

    # Display error message if workflow isn't defined
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    # Text input for user message
    st.text_input(
        "You: ", 
        st.session_state.user_input, 
        key="user_input", 
        on_change=process_input
    )



def teacher_main():
    st.write("### Teacher Section")

    def upload_answer_sheet():
        # result = tb.evaluate_answers(st.session_state.questions_list, st.session_state.answer_sheet, st.session_state.marks_list, st.session_state.text)
        result = tb.report_generator_using_llm(st.session_state.retriever,st.session_state.questions_list,st.session_state.answer_sheet,st.session_state.marks_list,st.session_state.text)
        # print(result)

        # st.write(result)
        st.text(result)
        

    # ---- Display all saved answers outside the function ----
    def display_written_answers():
        if "answer_sheet" in st.session_state:
            st.write("### Saved Answers:")
            for i, answers in enumerate(st.session_state.answer_sheet, start=1):
                st.write(f"**Question {i} Answers:**")
                for idx, ans in enumerate(answers, start=1):
                    st.write(f"{idx}. {ans}")


    # Function to generate the exam
    def exam_generator(num_questions):
        if "text" in st.session_state:  # Check if material (text) exists in session state
            try:
                questions_list,marks_list = tb.question_paper_generator(
                    st.session_state.retriever, 
                    text=st.session_state.text, 
                    questions=num_questions #give option to the user for selecting number of questions.
                )

                st.session_state.questions_list = questions_list[:num_questions]
                st.session_state.marks_list = marks_list[:num_questions]
                
                # For TESTING!!!!!!!!!!!!!!!
#                 st.session_state.questions_list = ['What are the key stages involved in full stack development and how do they '
#  'contribute to the creation of a complete web application?',
#  'What are the responsibilities of a full stack developer and what skills are '
#  'required to be proficient in the role?',
#  'Explain the concept of 3-tier application architecture and how it enhances '
#  'flexibility, maintainability, and reusability in software development.',
#  'Describe the key constraints of RESTful APIs and how they enable effective '
#  'communication between client and server.',
#  'What are the advantages of using JSON as a data interchange format in web '
#  'applications and how does it support data structures and serialization?']
#                 st.session_state.marks_list = [10,10,10,10,10]


                # Display all the generated questions
                total_marks = sum(st.session_state.marks_list)

            # Display total marks at the top
                # st.write("### Generated Exam Questions")
                # st.write(f"**Total Marks: {total_marks}**")
                # for i, (question, marks) in enumerate(zip(st.session_state.questions_list, st.session_state.marks_list), start=1):
                #     st.write(f"**{i}. {question}**")
                #     st.write(f"*Marks: {marks}*")

                # st.write("### Upload Answer Sheet")
                # st.write("Please provide answers for the following questions:")

                
                
            except Exception as e:
                st.error(f"An error occurred while generating the exam: {e}")
        else:
            # Show error if no text material is provided
            st.session_state.error_message = "Please provide the material!"
            st.error(st.session_state.error_message)


    def display_generated_questions():
        """Display generated exam questions and marks if they exist in session state."""
        if "questions_list" in st.session_state and "marks_list" in st.session_state:
            # Calculate total marks
            total_marks = sum(st.session_state.marks_list)

            # Display total marks and questions with marks
            st.write("### Generated Exam Questions")
            st.write(f"**Total Marks: {total_marks}**")

            # Iterate and display each question and its marks
            # for i, (question, marks) in enumerate(zip(st.session_state.questions_list, st.session_state.mark_list), start=1):
            #     st.write(f"**{i}. {question}**")
            #     st.write(f"*Marks: {marks}*")    


            # Initialize session_state for answers if not already present
            if "answer_sheet" not in st.session_state:
                st.session_state.answer_sheet = [[] for _ in range(len(st.session_state.questions_list))]

            # Display text areas and handle input appending
            for i, (question, marks) in enumerate(zip(st.session_state.questions_list, st.session_state.marks_list), start=1):
                st.write(f"**Question {i}: {question}**")
                st.write(f"*Marks: {marks}*")    
                new_answer = st.text_area(
                    f"Answer {i}", 
                    placeholder="Write your answer here and press Ctrl+Enter to save.",
                    height=150, 
                    key=f"answer_{i}"
                )

                # Append the text input to the list if it’s non-empty
                if new_answer and new_answer not in st.session_state.answer_sheet[i-1]:
                    st.session_state.answer_sheet[i-1].append(new_answer)        

    
    # num_questions = st.radio("Select the number of questions", [5, 10], index=0)
    # Button to trigger the exam generation
    if st.button("Exam Generator"):
        exam_generator(5)

    display_generated_questions()

    # Button to upload answer sheet
    if st.button("Upload Answer Sheet"):
        upload_answer_sheet()

def handle_user_input(input_text):
    # Custom logic goes here
    st.write(f"You entered: {input_text}")
    st.write("Custom function executed!")

# Main IIICell section
def iiicell_main():
    def handle_user_input(input_text):
        # For Production, uncomment this!!!!!!!!!
        response = st.session_state.iii_workflow.invoke({'input': input_text})
        final_dict = response['final_dict']

        role = user_input
        top_courses = iii.serper_tool(role)
        courses_with_links_last = []

        for course in top_courses:
            # We create a dictionary where we place the title first and the link last
            course_data = {
                'Title': course['title'],
                'Link': course['link']
            }
            courses_with_links_last.append(course_data)
        

        # Jobs Section
        st.header("Job Recommendations")
        job_data = pd.DataFrame({
            "Job Title": final_dict.job_title,
            "Company": final_dict.job_company,
            "Location": final_dict.job_location,
            "Job Link": final_dict.job_url
        })
        st.dataframe(job_data)
        courses_data = pd.DataFrame({
    "Course Title": [course['title'] for course in top_courses],
    "Course Link": [course['link'] for course in top_courses]
})

        # Display the header and the courses table
        st.header("Course Recommendations")
        st.dataframe(courses_data)
        # st.dataframe(course_data)

        # Optional: Display Links as Clickable (Better User Experience)
        st.write("### Jobs with Links")
        for title, url in zip(final_dict.job_title, final_dict.job_url):
            st.markdown(f"- [{title}]({url})")
        st.write("### Courses with Links")
        for course in top_courses:
            st.markdown(f"- [{course['title']}]({course['link']})")



    st.write("### IIICell Section")
    
    # Text input from the user
    user_input = st.text_input("Enter your text here:", key="iiicell_input")
    
    # Check if input exists and process it
    if user_input:
        if user_input.strip():
            handle_user_input(user_input)
        else:
            st.warning("Please enter some text before submitting.")

def chatbot():
    st.write("## General Bot")
    
    # Initialize session state variables if not present
    if "gen_user_input" not in st.session_state:
        st.session_state.gen_user_input = ""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "gen_messages" not in st.session_state:
        st.session_state.gen_messages = []

    SUMMARY_THRESHOLD = 15
    summary_prompt = "Summarize the following conversation very briefly:"

    def process_input():
        user_input = st.session_state.gen_user_input  # Retrieve user input
        
        if user_input:  # Only process if the input is not empty
            # Append user message to conversation history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})

            # Check if summarization is needed
            if len(st.session_state.conversation_history) > SUMMARY_THRESHOLD:
                summary_request = {"messages": [{"role": "user", "content": summary_prompt}] + st.session_state.conversation_history}
                summary_result = st.session_state.gen_bot.invoke(summary_request)
                summary = summary_result["messages"][-1].content
                # Replace history with summary and recent messages
                st.session_state.conversation_history = [
                    {"role": "system", "content": f"Summary of prior conversation: {summary}"}
                ] + st.session_state.conversation_history[-5:]

            # Process user input with the bot and get response
            conversation = {"messages": st.session_state.conversation_history}
            results = st.session_state.gen_bot.invoke(conversation)
            bot_response = results["messages"][-1].content

            # Append bot response to conversation history
            st.session_state.conversation_history.append({"role": "assistant", "content": bot_response})
            
            # Clear the input field
            st.session_state.gen_user_input = ""

    # Display conversation history
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        elif msg["role"] == "bot":
            st.markdown(f"**Bot:** {msg['content']}")
        else:
            st.markdown(f"**System:** {msg['content']}")

    # Text input for user message
    st.text_input(
        "You: ", 
        st.session_state.gen_user_input, 
        key="gen_user_input", 
        on_change=process_input
    )



def main():
    st.title('DocSpark')
    option = st.sidebar.radio("Choose a section:", ["Library", "Teacher", "IIICell","General_Bot"])

    # Initialize session state if not already set
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload multiple files", 
        type=["pdf", "mp3", "wav", "mp4"], 
        accept_multiple_files=True
    )

    # Compare and sync uploaded files (to handle the "X" cross behavior)
    if uploaded_files != st.session_state.uploaded_files:
        st.session_state.uploaded_files = uploaded_files

    # Display uploaded files
    if st.session_state.uploaded_files:
        # st.write("Uploaded files:")
        # for i, file in enumerate(st.session_state.uploaded_files):
        #     st.write(f"- {file.name}")
        
        # Process the files
        if st.sidebar.button("Add Files"):
            ans = handle_file_upload(st.session_state.uploaded_files)
            # st.write(ans)

    # Show current session state for debugging
    # st.write("Session State:", st.session_state.uploaded_files)
    # st.write("Number of files:", len(st.session_state.uploaded_files))
    # Dynamically render content based on the selected button
    if option == "Library":
        library_main()
    elif option == "Teacher":
        teacher_main()
    elif option == "IIICell":
        iiicell_main()
    elif option == "General_Bot":
        chatbot()

# Run the Streamlit app
if __name__ == "__main__":
    main()
