# 🤖 LLM Orchestrator

**LLM Orchestrator** is an open-source application that lets you interact with multiple large language models (LLMs) through an intuitive web interface built with **Streamlit**. It smartly picks the best model for your query, refines answers iteratively, and keeps your conversations flowing naturally. Whether you're coding, analyzing data, or just curious, **LLM Orchestrator** makes it easy and effective.

---

## 🚀 Key Features

- **Smart Model Selection**: Automatically picks the right LLM for your question (e.g., coding, analysis, or general queries).
- **Response Refinement**: Improves answers step-by-step using multiple models for top-notch results.
- **Chat History**: Saves your conversations and keeps the context intact.
- **Resource Awareness**: Adapts to your machine’s RAM and GPU to suggest suitable models.
- **Multilingual Magic**: Detects your question’s language and responds accordingly (bonus points for French with `mistral`).
- **User-Friendly Design**: A clean, simple interface powered by **Streamlit**.

---

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)**: Powers the interactive web interface.
- **[Ollama](https://ollama.ai/)**: Handles LLM integration and management.


---

## 📦 Installation and Setup

### Prerequisites
- **Python 3.8+** installed on your system.
- **Ollama** set up and running. Check out the [Ollama Installation Guide](https://ollama.ai/docs/installation).
- Download your preferred LLM models via Ollama (e.g., `ollama pull llama2`).

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-orchestrator.git
   cd llm-orchestrator
   ```


 2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
