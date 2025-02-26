🤖 LLM Orchestrator
LLM Orchestrator is an open-source application that lets you interact with multiple large language models (LLMs) through an intuitive web interface built with Streamlit. It smartly picks the best model for your query, refines answers iteratively, and keeps your conversations flowing naturally. Whether you're coding, analyzing data, or just curious, LLM Orchestrator makes it easy and effective.

🚀 Key Features
Smart Model Selection: Automatically picks the right LLM for your question (e.g., coding, analysis, or general queries).
Response Refinement: Improves answers step-by-step using multiple models for top-notch results.
Chat History: Saves your conversations and keeps the context intact.
Resource Awareness: Adapts to your machine’s RAM and GPU to suggest suitable models.
Multilingual Magic: Detects your question’s language and responds accordingly (bonus points for French with mistral).
User-Friendly Design: A clean, simple interface powered by Streamlit.
🛠️ Technologies Used
Streamlit: Powers the interactive web interface.
Ollama: Handles LLM integration and management.
📦 Installation and Setup
Prerequisites

Python 3.8+ installed on your system.
Ollama set up and running. Check out the Ollama Installation Guide.
Download your preferred LLM models via Ollama (e.g., ollama pull llama2).
Installation Steps

Clone the repository:
bash
Envelopper
Copier
git clone https://github.com/your-username/llm-orchestrator.git
cd llm-orchestrator
Install the required packages:
bash
Envelopper
Copier
pip install -r requirements.txt
Launch the app:
bash
Envelopper
Copier
streamlit run app.py
Open your browser and go to http://localhost:8501.
💬 How to Use It
Start a Chat: Hit "➕ New Chat" in the sidebar to kick off a new conversation.
Ask Away: Type your question in the text box and tweak the intelligence level (response speed) if you like.
Get Answers: The app picks the best model and shows you the response, along with the model used.
Switch Chats: Jump between past conversations using the sidebar.
Example:

Question: "How do I sort a list in Python?"
Response: A model like llama2 explains it step-by-step.
