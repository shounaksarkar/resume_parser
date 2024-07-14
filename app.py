import streamlit as st
import PyPDF2
from langchain_groq import ChatGroq
import re
import json
import io

def pdf_to_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def resume_formatter(llm, resume_text):
    prompt = f"""
    ## Instruction ##
    You are an AI assistant tasked with extracting and formatting key information from resumes. Given a resume text, extract and format the information for the following five fields:

    1. CONTACT INFORMATION
    2. WORK EXPERIENCE
    3. EDUCATION
    4. SKILLS
    5. PROFESSIONAL SUMMARY

    Format your response exactly as shown in the examples below, using all caps for field names and quotation marks for the content. Ensure that the information is accurate and concise.
    If the summary is not provided in the resume, you must summarize it yourself and provide.
    If any information relating to the resume is missing, you just give "NA" for that field. Whatever you answer must be from the given resume text only.

    Example 1:

    CONTACT INFORMATION:
    "Pathik Ghugare
    Mumbai, Maharashtra
    pathikghugare13@gmail.com
    8425817345
    GitHub
    LinkedIn"

    WORK EXPERIENCE:
    "Pibit.ai (YC 21) Bengaluru, Karnataka
    Machine Learning Engineer Jun 2023–Present
    - Played a crucial role in conceptualizing and developing the Loss Run product
    - Automated data preparation and model training pipelines on Azure Machine Learning
    - Conducted extensive experiments with YOLOv5 and YOLOv8 models
    - Deployed models as a service using various tools and platforms
    - Developed 3 identity services by integrating YOLO and GPT-3.5
    - Explored capabilities of GPT-4V for company's problem statements
    - Experimented with various prompting methodologies

    Pibit.ai (YC 21) Bengaluru, Karnataka
    Deep Learning Intern August 2022–Jun 2023
    - Enhanced existing object detection model (YOLOv5)
    - Integrated Azure MLFlow for experiment and model tracking
    - Trained an OCR-free document model (DONUT)
    - Developed and deployed multiple YOLOv5 models for identity services
    - Built an internal dashboard with Streamlit
    - Worked on template standardization"

    EDUCATION:
    "K.J.Somaiya College Of Engineering Mumbai, Maharashtra
    Computer Engineering, B.Tech 9.35 CGPA 2019–2023
    Coursework: Data Mining and Analysis, Software Engineering, Operating Systems, Artificial Intelligence"

    SKILLS:
    "Frameworks: HuggingFace, Langchain, Llamaindex, PyTorch, Tensorflow, Keras, Pandas, MLFlow, RAGAS
    Tools: Git, GitHub, VSCode/PyCharm, Cursor.sh, LMStudio, Ollama
    Infrastructure: Kubernetes, Helm, Docker
    Cloud Services: AWS, Azure, GCP, Runpod
    Databases: MySQL, PostgreSQL, DynamoDB"

    PROFESSIONAL SUMMARY:
    "Experienced Machine Learning Engineer with a strong background in deep learning, computer vision, and natural language processing. Proficient in developing and deploying AI models, with expertise in YOLO, GPT, and LLaMA architectures. Skilled in cloud technologies, particularly Azure and AWS, and experienced in automating ML pipelines. Demonstrated ability to improve model performance and efficiency in real-world applications. Strong academic background with a B.Tech in Computer Engineering and relevant coursework in AI and data mining."

    Example 2:

    CONTACT INFORMATION:
    "SHOUNAK SARKAR
    Email: sarkarshounak7@gmail.com
    Phone number: 8697941058"

    WORK EXPERIENCE:
    "AI Intern, Kreat.ai
    April 2024 – present (Bengaluru, Karnataka)
    - Scraped and processed datasets with over 50 million data points for storage in ChromaDB on Microsoft Azure
    - Developed 50+ advanced LLM prompts using CoT, in-context learning, few-shot, and format-preserving prompting
    - Created and tested APIs for knowledge graph and LLM integration
    - Utilized Scrapy, Langchain, Neo4j, GCP BigQuery, Postman, Streamlit, Azure OpenAI, FastAPI and CrewAI"

    EDUCATION:
    "BTECH IN ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING
    MCKV INSTITUTE OF ENGINEERING, HOWRAH (2021-25)
    CGPA: 9.6. Currently studying in 6th semester.

    TECHNO INDIA GROUP OF PUBLIC SCHOOL, NABAGRAM
    CBSE - CLASS 12: 89%

    GOSPEL HOME SCHOOL, RISHRA
    ICSE CLASS 10: 91%"

    SKILLS:
    "C, C++, Python
    HTML, CSS, Javascript, React.js
    Data Analysis (Excel, PowerBI)
    Machine learning (Regression, Classification, Clustering algorithms)
    Tensorflow: from data to deployment
    Langchain, LLamaIndex, LLM-RAG Application
    Knowledge-Graph Based LLM powered Recommendation Engine
    neo4j, Chainlit, CrewAI"

    PROFESSIONAL SUMMARY:
    "Ambitious AI and Machine Learning student with a strong academic background and practical experience in data processing, LLM technologies, and web development. Demonstrated leadership skills through various positions of responsibility and technical proficiency through high rankings in multiple hackathons. Possesses a diverse skill set including programming languages, data analysis tools, and machine learning frameworks. Currently gaining professional experience as an AI Intern, working with large-scale datasets and advanced LLM applications."

    Now, extract and format the information from the following resume text:

    {resume_text}
    """

    response = llm.invoke(prompt)
    return response.content

def parse_resume_output(output):
    # Initialize a dictionary to store the parsed values
    parsed_data = {
        "CONTACT INFORMATION": "",
        "WORK EXPERIENCE": "",
        "EDUCATION": "",
        "SKILLS": "",
        "PROFESSIONAL SUMMARY": ""
    }

    # Split the output by lines and process each line
    lines = output.split('\n')

    current_field = None

    for line in lines:
        line = line.strip()
        if line.startswith("CONTACT INFORMATION:"):
            current_field = "CONTACT INFORMATION"
            parsed_data[current_field] = line.split(":")[1].strip().strip('"')
        elif line.startswith("WORK EXPERIENCE:"):
            current_field = "WORK EXPERIENCE"
            parsed_data[current_field] = line.split(":")[1].strip().strip('"')
        elif line.startswith("EDUCATION:"):
            current_field = "EDUCATION"
            parsed_data[current_field] = line.split(":")[1].strip().strip('"')
        elif line.startswith("SKILLS:"):
            current_field = "SKILLS"
            parsed_data[current_field] = line.split(":")[1].strip().strip('"')
        elif line.startswith("PROFESSIONAL SUMMARY:"):
            current_field = "PROFESSIONAL SUMMARY"
            parsed_data[current_field] = line.split(":")[1].strip().strip('"')
        elif current_field:
            parsed_data[current_field] += " " + line.strip().strip('"')

    return parsed_data

st.set_page_config(page_title="Resume Parser",page_icon="💡")

st.title("Resume to JSON")

intro = """
This application converts PDF resumes into structured JSON format. It extracts and formats the following fields:

- CONTACT INFORMATION 
- WORK EXPERIENCE '
- EDUCATION '
- SKILLS 
- PROFESSIONAL SUMMARY 
 
Upload a PDF resume, and the app will parse it using natural language processing, providing a JSON output of the extracted information.
"""
st.markdown(intro)

llm = ChatGroq(temperature=0.5, groq_api_key="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", model_name="llama3-70b-8192")


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_file = io.BytesIO(uploaded_file.read())
    with st.spinner("Parsing your resume..."):
        text = pdf_to_text(pdf_file)
        llm_response = resume_formatter(llm,text)
        output = parse_resume_output(llm_response)

    st.markdown("### Extracted JSON")
    st.write(output)



