DocuChat: Your AI-Powered Document Assistant 🚀

Tired of manually digging through scanned PDFs and handwritten notes? Meet DocuChat – your intelligent document sidekick that reads, understands, and chats about your files! 📄✨🤖

DocuChat is an AI-powered assistant that allows you to upload documents (PDFs, images, even handwritten notes!) and interact with them via natural language queries. Get instant summaries, answers, key insights, and more – all powered by advanced text extraction and a Retrieval-Augmented Generation (RAG) pipeline.
This is an open-source project built as a portfolio piece to showcase full-stack development, AI integration, and document processing.

✨ Features

Text Extraction: Pulls content from PDFs and images, including OCR for handwritten or messy text ✍️.

Interactive Chat: Ask questions like "Summarize this document," "List the key points," or "What does this section mean?"

Multi-File Support: Handle multiple documents in a single session for comprehensive analysis.

Source References: Responses include citations to original document sections for transparency and verification.

User-Friendly Interface: Sleek, responsive design for easy uploads and conversations.

🛠 Built With

Frontend: React.js + Bootstrap (for a modern, responsive UI)

Backend: Python + FastAPI (for efficient API handling)

Text Processing: EasyOCR (OCR capabilities) & pdfplumber (PDF parsing)

AI Pipeline: RAG (Retrieval-Augmented Generation) for context-aware, accurate responses

Other Tools: Supports virtual environments for dependency management

📦 Getting Started

Prerequisites

Python 3.11

React.js (for frontend)

Git

Installation

Clone the Repository:

git clone https://github.com/harshhkalia/DocuChat-Your-AI-Summariser.git

cd DocuChat-Your-AI-Summariser

Set Up Backend (Python):

Create and activate a virtual environment (highly recommended to avoid conflicts!): python -m venv .venv

On Linux/Mac: source .venv/bin/activate

On Windows: .venv\Scripts\activate

Install dependencies:pip install -r requirements.txt

Set Up Frontend (React):
Navigate to the frontend directory (if structured as such, e.g., cd frontend): npm install

Run the Application:

Start the backend (e.g., with Uvicorn):uvicorn main:app --reload  # Assuming main.py is the entry point

Start the frontend: npm start

Open your browser at http://localhost:3000 (or the specified port) and start uploading documents!

🚨 Note: Always activate the virtual environment before running commands, or things might get weird! 😅 If you encounter issues, check for missing API keys (e.g., for any external AI services) in a .env file.

Usage

Upload your document (PDF, image, etc.) via the web interface.
Start chatting: Type queries about the content.
Get AI-generated responses with references back to the source.

Example Queries:
"Summarize the main ideas."
"Extract all dates mentioned."
"Tell me key features about the document"

🤝 Contributing
This is an open-source project, and I'd love your feedback, suggestions, or contributions! 

Star the repo if you find it useful 🌟.
Open issues for bugs or feature requests.
Submit pull requests for improvements.
Let's make DocuChat even smarter together!

👋 About the Author

Built by Harsh Kalia. Connect with me on LinkedIn for more tech discussions!  
https://www.linkedin.com/in/harsh-kalia-818990286/

#AI #DocumentAssistant #OCR #Python #React #FastAPI #OpenSource #Developer #Coding #MachineLearning #RAG #GitHub #Tech #PortfolioProject #DocuChat
