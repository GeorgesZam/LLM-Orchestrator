import streamlit as st
import ollama
import os
import psutil
import platform
import subprocess
from datetime import datetime
from PIL import Image
import io
import json
import time
import re

# Streamlit page configuration
st.set_page_config(page_title="LLM Orchestrator", layout="wide")

# Custom CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1rem;
    }
    .model-card {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        background: none;  /* Remove gray background */
    }
    .response-container {
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)


def get_gpu_memory():
    """Get available GPU memory in GB"""
    try:
        if platform.system() == "Darwin":  # macOS
            cmd = "system_profiler SPDisplaysDataType"
            output = subprocess.check_output(cmd.split()).decode()
            if "Metal" in output:
                return 4  # Estimation for Metal
            return 0
        elif platform.system() == "Windows":
            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
            try:
                output = subprocess.check_output(cmd.split())
                return int(output.decode().strip()) / 1024  # Convert to GB
            except:
                return 0
        elif platform.system() == "Linux":
            try:
                output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
                return int(output.decode().strip()) / 1024
            except:
                return 0
        return 0
    except:
        return 0


# Machine configuration
if 'MACHINE_CONFIG' not in st.session_state:
    st.session_state.MACHINE_CONFIG = {
        'max_model_size': 14,  # Maximum size in billions of parameters
        'ram_gb': round(psutil.virtual_memory().total / (1024 ** 3), 1),  # Total RAM in GB
        'gpu_memory_gb': get_gpu_memory()  # GPU memory in GB
    }


# Configuration interface
def show_machine_config():
    st.sidebar.markdown("### 🖥️ Machine Configuration")
    st.session_state.MACHINE_CONFIG['max_model_size'] = st.sidebar.number_input(
        "Max model size (B parameters)",
        min_value=1,
        max_value=70,
        value=st.session_state.MACHINE_CONFIG['max_model_size']
    )

    # Display detected resources
    st.sidebar.info(f"""
        💾 RAM: {st.session_state.MACHINE_CONFIG['ram_gb']} GB
        🎮 GPU: {st.session_state.MACHINE_CONFIG['gpu_memory_gb']} GB

        Recommendations:
        - 8GB RAM min. per 7B model
        - 16GB RAM min. per 13B+ model
        - GPU recommended for large models
    """)


def initialize_session_state():
    """Initialize session state variables"""
    if 'OLLAMA_HOST' not in st.session_state:
        st.session_state.OLLAMA_HOST = "http://localhost:11434"
    if 'client' not in st.session_state:
        st.session_state.client = None
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'chat_counter' not in st.session_state:
        st.session_state.chat_counter = 0
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


def get_available_models():
    """Get list of available models on the machine"""
    try:
        response = st.session_state.client.list()
        models = []

        for model in response.models:
            if hasattr(model, 'model'):
                name = str(model.model)
                # Ajouter tous les modèles, y compris ceux avec des tags spécifiques
                models.append(name)

        return models
    except Exception as e:
        st.error(f"Error retrieving models: {str(e)}")
        return []


def get_manager_prompt(question, max_time, available_models, files=None):
    """Create prompt for LLM manager"""
    file_info = ""
    if files:
        file_info = "\nAttached files:\n" + "\n".join([f"- {f.name} ({f.type})" for f in files])
    
    return f"""You are an expert in LLM model selection. Quickly choose the most appropriate model.

Question: "{question}"
{file_info}

Available models:
{json.dumps(list(available_models.values()), indent=2)}

Simple rules:
- Calculations, greetings: small model
- Knowledge questions: medium model
- Creative questions, suggestions: large model
- Complex analysis: large model
- Images: vision model if available
- Documents: tool model if available

JSON format only:
{{
    "selected_models": ["model_name"],
    "strategy": "Short reason"
}}"""


def get_manager_decision(question, reasoning_level):
    """Get quick decision from manager based on reasoning level and question"""
    available_models = get_available_models()

    if not available_models:
        st.error("No models available")
        return None

    # Check if images are present in the last message
    current_chat = st.session_state.chats.get(st.session_state.current_chat_id, {})
    messages = current_chat.get('messages', [])
    has_images = False
    
    if messages and 'files' in messages[-1]:
        has_images = any(file['type'].startswith('image/') for file in messages[-1]['files'])

    # If image is present, look for vision models
    if has_images:
        vision_models = [m for m in available_models 
                        if any(name in m.lower() for name in ['llava', 'bakllava', 'vision'])]
        if vision_models:
            return {
                'selected_models': [vision_models[0]],
                'strategy': "Using vision model for image analysis"
            }

    # Model selection based on reasoning level
    if reasoning_level == "Quick":
        preferred_models = ['mistral', 'phi']
    elif reasoning_level == "Thoughtful":
        preferred_models = ['llama2', 'mixtral']
    else:  # Deep
        preferred_models = ['mixtral', 'solar', 'llama2:70b']

    # Find the first available preferred model
    for model_type in preferred_models:
        matching_models = [m for m in available_models if model_type.lower() in m.lower()]
        if matching_models:
            return {
                'selected_models': [matching_models[0]],
                'strategy': f"Using {matching_models[0]} for {reasoning_level.lower()} reasoning"
            }

    # Fallback to any available model
    return {
        'selected_models': [available_models[0]],
        'strategy': "Using default model"
    }


def analyze_question_complexity(question):
    """Analyze question complexity"""
    simple_patterns = [
        r'^\d+[\s+\-*/]\d+',  # Simple math
        r'^(hi|hello|hey)\b',  # Greetings
        r'^what(\s+is)?\s+\d+[\s+\-*/]\d+',  # Math questions
    ]

    for pattern in simple_patterns:
        if re.match(pattern, question.lower()):
            return "SMALL"

    return None


def get_appropriate_model(available_models, question):
    """Select most appropriate model based on question"""
    complexity = analyze_question_complexity(question)
    if complexity == "SMALL":
        return list(available_models.values())[0]
    else:
        return list(available_models.values())[0]


def get_response(model, question, chat_id, force_web_search=False):
    """Get response with chat context"""
    try:
        response_container = st.empty()
        response_container.markdown(f"*🤖 Réponse de {model}...*")
        
        needs_search = force_web_search or any(keyword in question.lower() for keyword in [
            "actualité", "dernière", "récent", "nouveau", "news",
            "quoi de neuf", "qu'est-ce qui se passe"
        ])
        
        if needs_search:
            response_container.markdown("*🔎 Recherche web...*")
            search_results = search_brave(question)
            
            # Formatage des résultats pour le LLM
            formatted_results = "\n\n".join([
                f"Source: {r['title']}\n{r['description']}" 
                for r in search_results
            ])
            
            # Utilisation du modèle local pour résumer les résultats
            messages = [
                {'role': 'system', 'content': 'Vous êtes un assistant qui résume des informations web en français de manière concise.'},
                {'role': 'user', 'content': f"Résumez ces résultats de recherche sur '{question}':\n\n{formatted_results}"}
            ]
            
            stream_response = st.session_state.client.chat(
                model=model,
                messages=messages,
                stream=True
            )
            
            full_response = ""
            for chunk in stream_response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    full_response += content
                    response_container.markdown(f"*🤖 {model}:*\n{full_response}▌")
        else:
            # Préparation du message système
            messages = [
                {'role': 'system', 'content': 'Vous êtes un assistant intelligent capable d\'analyser des images et du texte. Répondez en français.'}
            ]

            # Préparation du message utilisateur avec l'image
            user_message = {
                'role': 'user',
                'content': question
            }

            # Si le dernier message contient des images
            if st.session_state.chats[chat_id]['messages'] and 'files' in st.session_state.chats[chat_id]['messages'][-1]:
                images = []
                for file in st.session_state.chats[chat_id]['messages'][-1]['files']:
                    if file['type'].startswith('image/'):
                        # Convertir l'image en bytes
                        image = Image.open(io.BytesIO(file['content']))
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        images.append(img_byte_arr.getvalue())
            
                if images:
                    user_message['images'] = images

            messages.append(user_message)

            # Appel au modèle avec gestion du streaming
            stream_response = st.session_state.client.chat(
                model=model,
                messages=messages,
                stream=True
            )

            full_response = ""
            for chunk in stream_response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    full_response += content
                    response_container.markdown(f"*🤖 {model}:*\n{full_response}▌")

            response_container.markdown(f"*🤖 {model}:*\n{full_response}")

        # Sauvegarder la réponse dans l'historique
        st.session_state.chats[chat_id]['messages'].append({
            'role': 'assistant',
            'content': full_response,
            'model': model,
            'web_search': needs_search
        })

        save_chats()
        return {'status': 'success', 'content': full_response}

    except Exception as e:
        st.error(f"Erreur : {str(e)}")
        return {'status': 'error', 'content': str(e)}


def create_new_chat():
    """Create a new chat"""
    chat_id = str(st.session_state.chat_counter)
    st.session_state.chats[chat_id] = {
        'title': f"New Chat {chat_id}",
        'messages': [],
        'model': None,
        'created_at': datetime.now().isoformat()
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_counter += 1
    save_chats()


def save_chats():
    """Save chats to JSON file"""
    try:
        chats_data = {
            chat_id: {
                'title': chat['title'],
                'messages': chat['messages'],
                'model': chat.get('model'),
                'created_at': chat['created_at']
            }
            for chat_id, chat in st.session_state.chats.items()
        }
        
        with open('chats.json', 'w', encoding='utf-8') as f:
            json.dump(chats_data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde des chats : {str(e)}")


def load_chats():
    """Load chats from JSON file"""
    try:
        if not os.path.exists('chats.json'):
            # Créer le fichier s'il n'existe pas
            with open('chats.json', 'w', encoding='utf-8') as f:
                json.dump({}, f)
            return
            
        with open('chats.json', 'r', encoding='utf-8') as f:
            chats_data = json.load(f)
            
        st.session_state.chats = chats_data
        
        # Mettre à jour le compteur de chat
        if chats_data:
            max_chat_id = max(int(chat_id) for chat_id in chats_data.keys())
            st.session_state.chat_counter = max_chat_id + 1
            
    except Exception as e:
        st.error(f"Erreur lors du chargement des chats : {str(e)}")
        # Créer un fichier vide en cas d'erreur
        with open('chats.json', 'w', encoding='utf-8') as f:
            json.dump({}, f)


def setup_ollama_connection():
    """Setup Ollama server connection"""
    st.sidebar.markdown("### 🖥️ Server Configuration")

    server_type = st.sidebar.radio(
        "Server Location:",
        ["Local", "Remote"],
        help="Choose where your Ollama server is running"
    )

    if server_type == "Local":
        host = st.sidebar.text_input(
            "Local Server URL",
            value="http://localhost:11434",
            help="Usually http://localhost:11434"
        )
    else:
        host = st.sidebar.text_input(
            "Remote Server URL",
            value="http://",
            help="Example: http://your-server:11434"
        )

    if st.sidebar.button("Connect to Server"):
        try:
            st.session_state.OLLAMA_HOST = host
            st.session_state.client = ollama.Client(host=host)
            st.sidebar.success("Connected to Ollama server!")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {str(e)}")
            st.session_state.client = None

    # Gestion des modèles si connecté
    if st.session_state.client:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🤖 Gestion des Modèles")
        
        # Liste des modèles disponibles
        available_models = get_available_models()
        if available_models:
            st.sidebar.markdown("**Modèles installés:**")
            for model in sorted(available_models):
                st.sidebar.markdown(f"- {model}")
        
        # Interface simplifiée pour ajouter un nouveau modèle
        st.sidebar.markdown("**Installer un nouveau modèle**")
        
        # Liste prédéfinie de modèles populaires
        popular_models = [
            "mistral",
            "llava",
            "llava:13b",
            "llama2",
            "llama2:13b",
            "mixtral",
            "codellama",
            "phi",
            "solar"
        ]
        
        # Menu déroulant pour les modèles populaires
        new_model = st.sidebar.selectbox(
            "Choisir un modèle à installer",
            [""] + popular_models,
            format_func=lambda x: "Sélectionner un modèle" if x == "" else x
        )
        
        # Bouton d'installation
        if new_model and st.sidebar.button("Installer le modèle", key="install_model"):
            with st.sidebar.spinner(f"Installation de {new_model}..."):
                try:
                    st.session_state.client.pull(new_model)
                    st.sidebar.success(f"✅ {new_model} installé avec succès!")
                    time.sleep(2)  # Petit délai pour voir le message
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Erreur lors de l'installation: {str(e)}")


def show_chat_sidebar():
    """Show sidebar with chat list"""
    st.sidebar.markdown("### Your Chats")

    if st.sidebar.button("New Chat"):
        create_new_chat()
        st.rerun()

    st.sidebar.markdown("---")

    for chat_id, chat in sorted(st.session_state.chats.items(), key=lambda x: x[1]['created_at'], reverse=True):
        col1, col2 = st.sidebar.columns([4, 1])

        title = chat['title']
        if chat['messages']:
            first_msg = chat['messages'][0]['content']
            title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg

        if col1.button(f"📝 {title}", key=f"chat_{chat_id}"):
            st.session_state.current_chat_id = chat_id
            st.rerun()


def show_current_chat():
    """Show current chat"""
    if not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chats:
        if st.session_state.chats:
            latest_chat_id = max(st.session_state.chats.keys())
            st.session_state.current_chat_id = latest_chat_id
        else:
            create_new_chat()
            st.rerun()
            return

    chat = st.session_state.chats[st.session_state.current_chat_id]

    if chat.get('model'):
        st.markdown(f"*🤖 Current model: {chat['model']}*")

    for msg in chat['messages']:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
            if 'files' in msg:
                for file in msg['files']:
                    if file['type'].startswith('image/'):
                        st.image(file['content'], caption=file['name'])
                    elif isinstance(file['content'], pd.DataFrame):
                        st.dataframe(file['content'])
                    else:
                        st.text(f"File: {file['name']}")
        else:
            model_name = msg.get('model', 'Unknown')
            st.markdown(f"**Assistant ({model_name}):** {msg['content']}")


def process_uploaded_files(files):
    """Process uploaded files and return their content"""
    file_contents = []
    for file in files:
        content = None
        try:
            if file.type.startswith('image/'):
                # Lire directement les bytes du fichier
                content = file.getvalue()
            elif file.type == 'text/csv':
                content = pd.read_csv(file)
            elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                content = pd.read_excel(file)
            elif file.type == 'text/plain':
                content = file.getvalue().decode()
            
            if content is not None:
                file_contents.append({
                    'name': file.name,
                    'type': file.type,
                    'content': content
                })
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier {file.name}: {str(e)}")
    
    return file_contents


def initialize_search_agent():
    """Initialise l'agent de recherche web avec DuckDuckGo"""
    try:
        search_tool = DuckDuckGoSearchRun()
        llm = ChatOllama(
            model="mistral",
            base_url=st.session_state.OLLAMA_HOST
        )
        
        agent = initialize_agent(
            tools=[search_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        return agent
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de l'agent de recherche : {str(e)}")
        return None


def search_brave(query):
    """Effectue une recherche web via Brave Search"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    url = f"https://search.brave.com/search?q={query.replace(' ', '+')}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for result in soup.select(".snippet"):
            title_elem = result.select_one(".title")
            url_elem = result.select_one(".url")
            description_elem = result.select_one(".description")

            if title_elem and url_elem:
                title = title_elem.text.strip()
                url = url_elem.text.strip()
                description = description_elem.text.strip() if description_elem else ""
                results.append({
                    "title": title,
                    "url": url,
                    "description": description
                })

        return results[:3]  # Retourne les 3 premiers résultats
        
    except Exception as e:
        return [{"title": "Erreur de recherche", "description": str(e)}]


def main():
    """Main entry point"""
    st.markdown("<h1 class='main-header'>LLM Orchestrator</h1>", unsafe_allow_html=True)

    initialize_session_state()
    setup_ollama_connection()
    load_chats()

    show_chat_sidebar()

    if not st.session_state.client:
        st.warning("Please configure and connect to an Ollama server first")
        return

    show_current_chat()

    if st.session_state.current_chat_id:
        # File upload for images only
        uploaded_files = st.file_uploader(
            "Upload images",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg']
        )

        # Update session state immediately
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
        # Check for images
        has_images = uploaded_files and any(file.type.startswith('image/') for file in uploaded_files)
        
        # Show vision model selector if images are present
        selected_model = None
        if has_images:
            available_models = get_available_models()
            llava_models = [m for m in available_models if 'llava' in m.lower()]
            
            if llava_models:
                st.markdown("### 🖼️ Image Detected - Vision Model Selection")
                selected_model = st.selectbox(
                    "Choose vision model:",
                    sorted(llava_models),
                    format_func=lambda x: f"🤖 {x}"
                )
                
                if selected_model:
                    st.info(f"Selected model: {selected_model}")
            else:
                st.warning("⚠️ No vision models available. Please install one with 'ollama pull llava'")

        question = st.text_area("Enter your message:", height=100)

        # New layout with 2 columns
        col1, col2 = st.columns([3, 1])
        with col1:
            reasoning_level = st.radio(
                "Reasoning level:",
                ["Quick", "Thoughtful", "Deep"],
                horizontal=True,
                help="Choose the model's reasoning level"
            )
            
        with col2:
            send_button = st.button("Send")

        if send_button:
            if question or uploaded_files:
                with st.spinner("Processing..."):
                    processed_files = process_uploaded_files(uploaded_files) if uploaded_files else None
                    
                    if has_images and selected_model:
                        decision = {
                            'selected_models': [selected_model],
                            'strategy': "Using selected vision model for image analysis"
                        }
                    else:
                        decision = get_manager_decision(question, reasoning_level)

                    if decision:
                        st.markdown("### Strategy:")
                        st.markdown(decision['strategy'])

                        for model_name in decision['selected_models']:
                            if processed_files:
                                st.session_state.chats[st.session_state.current_chat_id]['messages'].append({
                                    'role': 'user',
                                    'content': question,
                                    'files': processed_files,
                                    'model': model_name
                                })
                            
                            response = get_response(
                                model_name,
                                question,
                                st.session_state.current_chat_id
                            )


if __name__ == "__main__":
    main()
