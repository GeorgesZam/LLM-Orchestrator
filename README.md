# 🤖 LLM Orchestrator

**LLM Orchestrator** is an open-source application that lets you interact with multiple large language models (LLMs) through an intuitive web interface built with **Streamlit**. It smartly picks the best model for your query, refines answers iteratively, and keeps your conversations flowing naturally. Whether you're coding, analyzing data, or just curious, **LLM Orchestrator** makes it easy and effective.

---

## 🚀 Key Features

- **Smart Model Selection**: Automatically picks the right LLM for your question (e.g., coding, analysis, or general queries).
- **Response Refinement**: Improves answers step-by-step using multiple models for top-notch results.
- **Chat History**: Saves your conversations and keeps the context intact.
- **Resource Awareness**: Adapts to your machine's RAM and GPU to suggest suitable models.
- **Multilingual Magic**: Detects your question's language and responds accordingly (bonus points for French with `mistral`).
- **User-Friendly Design**: A clean, simple interface powered by **Streamlit**.

---

## 💡 Examples

### 1. Image Analysis
```bash
# First, install the vision model
ollama pull llava

# Then in the interface:
1. Upload an image
2. Select llava model from the dropdown menu
3. Ask a question like "What do you see in this image?"
```

### 2. Real-Time Web Search
```bash
# Use the "Web Search" checkbox for current events
Q: "What are the latest developments in AI?"
➡️ The application will automatically use DuckDuckGo to fetch recent information
```

### 3. Multilingual Conversation
```bash
# The application automatically detects the language
Q: "What is the weather like today?"
A: *Response in English*

Q: "Quel temps fait-il aujourd'hui ?"
A: *Response in French*
```

### 4. Image Analysis
```bash
1. Upload your file
2. Ask questions about its content
Q: "Can you analyze this picture"
```

### 5. Advanced Reasoning Levels
```bash
# Choose your reasoning level:
⚡ Fast: Quick responses for simple queries
🤔 Thoughtful: Balanced analysis for moderate complexity
🧠 Deep: Comprehensive analysis for complex questions

Example:
Q: "What are the implications of quantum computing on cybersecurity?"
Select: 🧠 Deep
```

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
 3. Launch the app:
    ```bash
    streamlit run app.py
    ```
 4. Launch ollama:
    ```bash
    ollama serve  
    ```
