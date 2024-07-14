```markdown
# Resume to JSON Converter

This Streamlit web application converts resumes in PDF format into structured JSON data using Langchain Groq for natural language processing.

## Features

- **Upload PDF**: Allows users to upload a resume in PDF format.
- **Parsing**: Extracts text content from the uploaded PDF resume.
- **Formatting**: Uses a language model to format extracted information into structured JSON.
- **Display**: Displays the formatted JSON output on the web interface.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:

   ```bash
   streamlit run app.py
   ```

2. Upload a PDF resume using the file uploader.
3. The application will parse the resume, format the extracted information into JSON, and display it on the web interface.

## Dependencies

- `streamlit`: For building and running the web application.
- `PyPDF2`: For reading text from PDF files.
- `langchain-groq`: For integrating with the Langchain Groq API for natural language processing.
