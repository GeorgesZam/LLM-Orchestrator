import streamlit as st
import ollama
import networkx as nx
from pyvis.network import Network
import pandas as pd
import json
import time
from streamlit_chat import message
import os
import re
import requests
import psutil
import platform
import subprocess
from datetime import datetime

# Dictionnaire des modèles et leurs caractéristiques
MODEL_INFO = {
    "llama2": {
        "size": "7B",
        "benchmarks": {
            "mmlu": 45.3,
            "arc": 42.1,
            "hellaswag": 76.4,
            "truthfulqa": 39.5
        },
        "specialties": ["general", "coding", "math"],
        "context_length": 4096
    },
    "llama2-13b": {
        "size": "13B",
        "benchmarks": {
            "mmlu": 54.8,
            "arc": 52.7,
            "hellaswag": 83.6,
            "truthfulqa": 41.2
        },
        "specialties": ["general", "analysis", "writing"],
        "context_length": 4096
    },
    "codellama-7b": {
        "size": "7B",
        "benchmarks": {
            "humaneval": 18.2,
            "mbpp": 28.3
        },
        "specialties": ["coding", "technical"],
        "context_length": 8192
    },
    "mistral": {
        "size": "7B",
        "benchmarks": {
            "mmlu": 62.5,
            "arc": 57.3,
            "hellaswag": 82.8
        },
        "specialties": ["general", "reasoning"],
        "context_length": 8192
    },
    "phi4": {
        "size": "14B",
        "benchmarks": {
            "mmlu": 55.0,
            "arc": 50.0,
            "hellaswag": 80.0
        },
        "specialties": ["general", "coding"],
        "context_length": 8192
    },
    "qwen2": {
        "size": "0.5B",
        "benchmarks": {
            "mmlu": 40.0,
            "arc": 35.0,
            "hellaswag": 70.0
        },
        "specialties": ["general"],
        "context_length": 4096
    }
}

# Configuration de la page Streamlit
st.set_page_config(page_title="LLM Manager Pro", layout="wide")

# Styles CSS personnalisés
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
        background: none;  /* Suppression du fond gris */
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
    """Obtient la mémoire GPU disponible en GB"""
    try:
        if platform.system() == "Darwin":  # macOS
            cmd = "system_profiler SPDisplaysDataType"
            output = subprocess.check_output(cmd.split()).decode()
            if "Metal" in output:
                return 4  # Estimation pour Metal
            return 0
        elif platform.system() == "Windows":
            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
            try:
                output = subprocess.check_output(cmd.split())
                return int(output.decode().strip()) / 1024  # Convertir en GB
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


# Configuration machine
if 'MACHINE_CONFIG' not in st.session_state:
    st.session_state.MACHINE_CONFIG = {
        'max_model_size': 14,  # Taille maximale en milliards de paramètres
        'ram_gb': round(psutil.virtual_memory().total / (1024 ** 3), 1),  # RAM totale en GB
        'gpu_memory_gb': get_gpu_memory()  # GPU memory en GB
    }


# Interface de configuration
def show_machine_config():
    st.sidebar.markdown("### 🖥️ Configuration Machine")
    st.session_state.MACHINE_CONFIG['max_model_size'] = st.sidebar.number_input(
        "Taille max. modèle (B paramètres)",
        min_value=1,
        max_value=70,
        value=st.session_state.MACHINE_CONFIG['max_model_size']
    )

    # Afficher les ressources détectées
    st.sidebar.info(f"""
        💾 RAM: {st.session_state.MACHINE_CONFIG['ram_gb']} GB
        🎮 GPU: {st.session_state.MACHINE_CONFIG['gpu_memory_gb']} GB

        Recommandations:
        - 8GB RAM min. par modèle de 7B
        - 16GB RAM min. par modèle de 13B+
        - GPU recommandé pour les gros modèles
    """)


# Ajout d'une fonction pour vérifier les modèles disponibles
def get_available_models():
    """Récupère la liste des modèles disponibles sur le serveur Ollama"""
    try:
        response = st.session_state.client.list()
        available_models = {}

        # Extraire les noms de modèles
        for model_obj in response.models:
            if hasattr(model_obj, 'model'):
                full_name = str(model_obj.model)  # ex: "llama2:latest"
                base_name = full_name.split(':')[0]  # ex: "llama2"

                # Vérifier si le modèle est dans notre configuration
                if base_name in MODEL_INFO:
                    available_models[base_name] = full_name

        if not available_models:
            st.error("Aucun modèle configuré n'est disponible sur le serveur Ollama")
            st.info("Modèles configurés : " + ", ".join(MODEL_INFO.keys()))
            st.info("Modèles installés : " + ", ".join(m.model for m in response.models if hasattr(m, 'model')))

        return available_models

    except Exception as e:
        st.error(f"Erreur lors de la récupération des modèles : {str(e)}")
        return {}


def select_best_models(question, max_time=60):
    """Sélectionne les modèles les plus pertinents selon la question"""
    available_models = get_available_models()

    # Détection de la langue
    is_french = any(word in question.lower() for word in ['é', 'è', 'ê', 'à', 'ç'])

    # Détection du type de question
    is_code = any(word in question.lower() for word in ['code', 'programming', 'python', 'javascript'])
    is_math = any(word in question.lower() for word in ['math', 'calcul', 'équation'])
    is_analysis = any(word in question.lower() for word in ['analyse', 'compare', 'différence'])

    # Score des modèles selon leurs spécialités
    model_scores = {}
    for model, info in MODEL_INFO.items():
        if model not in available_models:
            continue

        score = 0
        specialties = info['specialties']
        benchmarks = info['benchmarks']

        # Bonus pour Mistral en français
        if is_french and model == "mistral":
            score += 3

        # Bonus pour les spécialités
        if is_code and "coding" in specialties:
            score += 2
        if is_math and "math" in specialties:
            score += 2
        if is_analysis and "analysis" in specialties:
            score += 2

        # Bonus pour les benchmarks pertinents
        if "mmlu" in benchmarks:
            score += benchmarks["mmlu"] / 100

        model_scores[model] = score

    # Sélectionner les 2 meilleurs modèles
    best_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:2]

    return [(available_models[model], score) for model, score in best_models]


def iterative_improvement(question, models, max_time=60):
    """Améliore itérativement la réponse en utilisant plusieurs modèles"""
    start_time = time.time()
    best_response = None
    iteration = 0

    while time.time() - start_time < max_time:
        current_prompt = question if iteration == 0 else f"""
        Question originale: {question}
        Meilleure réponse actuelle: {best_response}
        Comment pouvons-nous améliorer cette réponse? Soyez plus précis et ajoutez des informations pertinentes.
        """

        for model_name, score in models:
            try:
                response = ""
                for chunk in get_response(model_name, current_prompt):
                    response += chunk

                if best_response is None or len(response) > len(best_response):
                    best_response = response

            except Exception as e:
                st.error(f"Erreur avec {model_name}: {e}")

        iteration += 1
        st.session_state.current_iteration = iteration

    return best_response


def get_response(model, question, chat_id):
    """Obtient la réponse avec contexte du chat"""
    try:
        # Afficher d'abord le modèle utilisé
        response_container = st.empty()
        response_container.markdown(f"*🤖 Réponse de {model}...*")

        full_response = ""

        # Construire le contexte
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant. Maintain context of the conversation.'}
        ]

        # Ajouter l'historique du chat actuel
        chat = st.session_state.chats[chat_id]
        for msg in chat['messages']:
            messages.append(msg)

        # Ajouter la nouvelle question
        messages.append({'role': 'user', 'content': question})

        stream_response = st.session_state.client.chat(
            model=model,
            messages=messages,
            stream=True
        )

        for chunk in stream_response:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                full_response += content
                response_container.markdown(f"*🤖 {model}:*\n{full_response}▌")

        # Afficher la réponse finale avec le modèle
        response_container.markdown(f"*🤖 {model}:*\n{full_response}")

        # Mettre à jour le chat
        chat['messages'].append({
            'role': 'user',
            'content': question,
            'model': model
        })
        chat['messages'].append({
            'role': 'assistant',
            'content': full_response,
            'model': model
        })
        chat['model'] = model

        # Sauvegarder
        save_chats()

        return {'status': 'success', 'content': full_response}

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {'status': 'error', 'content': str(e)}


def synthesize_responses(responses):
    """Crée une synthèse des réponses valides"""
    valid_responses = []
    skipped_models = []
    delire_models = []

    for r in responses:
        if r['status'] == 'success':
            valid_responses.append(r['content'])
        elif r['status'] == 'skipped':
            skipped_models.append(r['model'])
        elif r['status'] == 'delire':
            delire_models.append(r['model'])

    synthesis = ""

    if skipped_models:
        synthesis += f"*Modèles ignorés: {', '.join(skipped_models)}*\n\n"
    if delire_models:
        synthesis += f"*Modèles marqués comme délirants: {', '.join(delire_models)}*\n\n"

    if not valid_responses:
        return synthesis + "Aucune réponse valide n'a été obtenue."

    if len(valid_responses) == 1:
        return synthesis + valid_responses[0]

    synthesis += "### Synthèse des réponses:\n\n"
    for i, response in enumerate(valid_responses, 1):
        synthesis += f"**Point {i}:** {response.strip()}\n\n"

    return synthesis


def get_model_base_name(full_name):
    """Extrait le nom de base d'un modèle à partir de son nom complet"""
    return full_name.split(':')[0]


def get_model_categories(available_models, MODEL_INFO):
    """Détermine les catégories de modèles selon la machine"""
    try:
        # Récupérer les specs machine
        ram_gb = st.session_state.MACHINE_CONFIG.get('ram_gb', 8)  # 8GB par défaut
        gpu_memory_gb = st.session_state.MACHINE_CONFIG.get('gpu_memory_gb', 0)

        # Calculer la mémoire totale disponible
        total_memory = max(ram_gb, gpu_memory_gb if gpu_memory_gb else 0)

        # Adapter les seuils selon la mémoire disponible
        if total_memory >= 32:  # Machine puissante
            thresholds = {'GRAND': 13, 'MOYEN': 7}
        elif total_memory >= 16:  # Machine moyenne
            thresholds = {'GRAND': 7, 'MOYEN': 3}
        else:  # Machine modeste
            thresholds = {'GRAND': 3, 'MOYEN': 1}

        # Catégoriser les modèles disponibles
        categories = {}
        for name in available_models.keys():
            if name in MODEL_INFO:
                size = float(MODEL_INFO[name]['size'].replace('B', ''))
                if size >= thresholds['GRAND']:
                    categories[name] = 'GRAND'
                elif size >= thresholds['MOYEN']:
                    categories[name] = 'MOYEN'
                else:
                    categories[name] = 'PETIT'

        return categories, thresholds

    except Exception as e:
        # Valeurs par défaut si erreur
        default_categories = {
            name: 'MOYEN' if float(MODEL_INFO[name]['size'].replace('B', '')) >= 1 else 'PETIT'
            for name in available_models.keys() if name in MODEL_INFO
        }
        return default_categories, {'GRAND': 7, 'MOYEN': 1}


def get_manager_prompt(question, max_time, available_models, MODEL_INFO):
    """Crée le prompt pour le LLM manager en français"""
    return f"""Tu es un expert en sélection de modèles LLM. Choisis rapidement le modèle le plus adapté.

Question : "{question}"

Modèles disponibles :
{json.dumps(list(available_models.values()), indent=2)}

Règles simples :
- Calculs, salutations : petit modèle
- Questions de connaissance : modèle moyen
- Analyses complexes : grand modèle

Format JSON uniquement :
{{
    "selected_models": ["nom_du_modele"],
    "strategy": "Raison courte"
}}"""


def analyze_question_complexity(question):
    """Analyse la complexité de la question"""
    # Mots clés pour questions simples
    simple_patterns = [
        r'^hey\b',
        r'^hello\b',
        r'^hi\b',
        r'^\d+[\s+\-*/]\d+',  # Opérations mathématiques simples
        r'^bonjour\b',
        r'^salut\b',
    ]

    # Vérifier les patterns simples
    for pattern in simple_patterns:
        if re.match(pattern, question.lower()):
            return "PETIT"

    # Analyser la longueur et la complexité
    words = question.split()
    if len(words) < 10 and not any(char in question for char in '?!,;:'):
        return "PETIT"

    return None  # Laisser le manager décider


def get_manager_decision(question, max_time):
    """Obtient rapidement la décision du manager"""
    available_models = get_available_models()

    if not available_models:
        st.error("No models available")
        return None

    # Utiliser le plus petit modèle comme manager
    manager_model = min(
        available_models.values(),
        key=lambda m: float(MODEL_INFO[m.split(':')[0]]['size'].replace('B', ''))
    )

    try:
        response = st.session_state.client.chat(
            model=manager_model,
            messages=[{
                'role': 'user',
                'content': get_manager_prompt(question, max_time, available_models, MODEL_INFO)
            }],
            stream=False,
            options={'temperature': 0}  # Réponse plus directe
        )

        content = response['message']['content'].strip()
        if content.startswith("```"):
            content = re.sub(r'^```.*\n|```$', '', content)

        decision = json.loads(content)
        selected_model = decision['selected_models'][0]

        if selected_model in available_models.values():
            return decision
        else:
            fallback_model = list(available_models.values())[0]
            return {
                'selected_models': [fallback_model],
                'strategy': f"Modèle par défaut"
            }

    except Exception as e:
        st.error(f"Error: {str(e)}")
        fallback_model = list(available_models.values())[0]
        return {
            'selected_models': [fallback_model],
            'strategy': "Modèle par défaut (erreur)"
        }


def get_appropriate_model(available_models, question):
    """Sélectionne le modèle le plus approprié selon la question"""
    complexity = analyze_question_complexity(question)
    if complexity == "PETIT":
        # Trouver le plus petit modèle
        return min(
            available_models.values(),
            key=lambda m: MODEL_INFO[m.split(':')[0]]['size']
        )
    else:
        # Par défaut, prendre un modèle moyen
        return list(available_models.values())[0]


def load_chat_history():
    """Load chat history from JSON file"""
    history_file = "chat_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            st.session_state.chat_history = json.load(f)
    else:
        st.session_state.chat_history = []


def initialize_session_state():
    """Initialize session state variables"""
    if 'OLLAMA_HOST' not in st.session_state:
        st.session_state.OLLAMA_HOST = "http://localhost:11434"
    if 'client' not in st.session_state:
        st.session_state.client = None
    if 'chats' not in st.session_state:
        st.session_state.chats = {}  # {chat_id: {title, messages, model}}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'chat_counter' not in st.session_state:
        st.session_state.chat_counter = 0


# Déplacer l'initialisation après la définition des fonctions
if 'chat_history' not in st.session_state:
    load_chat_history()
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0
if 'best_response' not in st.session_state:
    st.session_state.best_response = None
if 'client' not in st.session_state:
    st.session_state.client = ollama.Client(host="http://localhost:11434")
if 'stop_generation' not in st.session_state:
    st.session_state.stop_generation = False


def save_chat_history(question, decision, responses):
    """Save chat to history"""
    chat_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'strategy': decision['strategy'],
        'models_used': [str(m) for m in decision['selected_models']],
        'responses': responses,
    }

    st.session_state.chat_history.append(chat_entry)

    # Save to file
    with open("chat_history.json", 'w', encoding='utf-8') as f:
        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)


def clear_model_memory(model_name):
    """Clear model's memory in Ollama"""
    try:
        st.session_state.client.delete(model_name)
        st.session_state.client.pull(model_name)
    except Exception as e:
        st.warning(f"Could not reset model {model_name}: {str(e)}")


def show_chat_history():
    """Display chat history with options"""
    st.sidebar.markdown("### 📚 Chat History")

    if not st.session_state.chat_history:
        st.sidebar.info("No chat history yet")
        return

    # Search in history
    search_term = st.sidebar.text_input("🔍 Search in history", "")

    # Filter history
    filtered_history = st.session_state.chat_history
    if search_term:
        filtered_history = [
            chat for chat in st.session_state.chat_history
            if search_term.lower() in chat['question'].lower()
        ]

    # Clear history button
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        os.remove("chat_history.json")
        st.sidebar.success("History cleared!")
        return

    # Display chats
    for i, chat in enumerate(filtered_history):
        with st.sidebar.expander(f"💭 {chat['question'][:50]}...", expanded=False):
            st.write(f"**Time:** {chat['timestamp']}")
            st.write(f"**Strategy:** {chat['strategy']}")

            # Copy question button
            if st.button("📋 Copy Question", key=f"copy_{i}"):
                st.session_state.copied_question = chat['question']
                st.success("Question copied!")

            # Show responses
            for model, response in chat['responses'].items():
                st.markdown(f"**Response from {model}:**")
                st.write(response)

                # Copy response button
                if st.button("📋 Copy Response", key=f"copy_response_{i}_{model}"):
                    st.session_state.copied_response = response
                    st.success("Response copied!")
                st.markdown("---")


def setup_ollama_connection():
    """Setup Ollama server connection"""
    st.sidebar.markdown("### ⚙️ Server Configuration")

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
            st.session_state.client.list()
            st.sidebar.success("✅ Connected to Ollama server!")
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {str(e)}")
            st.session_state.client = None


def show_conversation():
    """Affiche la conversation en cours avec contexte"""
    st.markdown("### 💬 Current Conversation")

    for msg in st.session_state.conversation_history:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant ({msg.get('model', 'Unknown')}):** {msg['content']}")

    # Afficher le modèle actuel
    if st.session_state.current_model:
        st.markdown(f"*Currently chatting with: {st.session_state.current_model}*")


def create_new_chat():
    """Crée un nouveau chat"""
    chat_id = str(st.session_state.chat_counter)
    st.session_state.chats[chat_id] = {
        'title': f"New Chat {chat_id}",
        'messages': [],
        'model': None,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_counter += 1
    save_chats()  # Sauvegarder immédiatement
    return chat_id


def save_chats():
    """Sauvegarde tous les chats dans un fichier"""
    with open("chats.json", 'w', encoding='utf-8') as f:
        json.dump(st.session_state.chats, f, ensure_ascii=False, indent=2)


def load_chats():
    """Charge les chats depuis le fichier"""
    if os.path.exists("chats.json"):
        with open("chats.json", 'r', encoding='utf-8') as f:
            st.session_state.chats = json.load(f)
            # Mettre à jour le compteur
            if st.session_state.chats:
                st.session_state.chat_counter = max(
                    int(chat_id) for chat_id in st.session_state.chats.keys()
                ) + 1


def show_chat_sidebar():
    """Affiche la sidebar avec la liste des chats"""
    st.sidebar.markdown("### 💭 Your Chats")

    # Bouton nouveau chat
    if st.sidebar.button("➕ New Chat"):
        create_new_chat()
        st.rerun()

    st.sidebar.markdown("---")

    # Liste des chats
    for chat_id, chat in sorted(st.session_state.chats.items(), key=lambda x: x[1]['created_at'], reverse=True):
        col1, col2 = st.sidebar.columns([4, 1])

        # Titre du chat avec modèle
        title = chat['title']
        if chat['messages']:
            first_msg = chat['messages'][0]['content']
            title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
            if chat.get('model'):
                title = f"{title} ({chat['model']})"

        # Sélection du chat
        if col1.button(f"📝 {title}", key=f"chat_{chat_id}"):
            st.session_state.current_chat_id = chat_id
            st.rerun()

        # Bouton suppression
        if col2.button("🗑️", key=f"del_{chat_id}"):
            if chat_id == st.session_state.current_chat_id:
                # Si on supprime le chat actuel, sélectionner le plus récent
                remaining_chats = [cid for cid in st.session_state.chats.keys() if cid != chat_id]
                st.session_state.current_chat_id = remaining_chats[0] if remaining_chats else None
            del st.session_state.chats[chat_id]
            save_chats()
            st.rerun()


def show_current_chat():
    """Affiche le chat actuel"""
    if not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chats:
        if st.session_state.chats:
            latest_chat_id = max(st.session_state.chats.keys())
            st.session_state.current_chat_id = latest_chat_id
        else:
            create_new_chat()
            st.rerun()
            return

    chat = st.session_state.chats[st.session_state.current_chat_id]

    # Afficher le modèle actuel si défini
    if chat.get('model'):
        st.markdown(f"*🤖 Current model: {chat['model']}*")

    # Afficher les messages
    for msg in chat['messages']:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
        else:
            model_name = msg.get('model', 'Unknown')
            st.markdown(f"**Assistant ({model_name}):** {msg['content']}")


def main():
    """Point d'entrée principal"""
    st.markdown("<h1 class='main-header'>🤖 LLM Manager Pro</h1>", unsafe_allow_html=True)

    initialize_session_state()
    setup_ollama_connection()
    load_chats()

    show_chat_sidebar()

    if not st.session_state.client:
        st.warning("⚠️ Please configure and connect to an Ollama server first")
        return

    show_current_chat()

    if st.session_state.current_chat_id:
        question = st.text_area("Enter your message:", height=100)

        col1, col2 = st.columns([3, 1])
        with col1:
            max_time = st.slider(
                "Response Intelligence Level",
                min_value=30,
                max_value=180,
                value=60,
                help="Higher IQ = more thoughtful responses"
            )
        with col2:
            st.markdown(f"""
            🧠 Intelligence:
            - IQ 100 (30)
            - IQ 120 (60)
            - IQ 140 (120)
            - IQ 160 (180)
            """)

        if st.button("Send"):
            if question:
                # Toujours demander une nouvelle décision
                with st.spinner("🤔 Analyzing..."):
                    decision = get_manager_decision(question, max_time)

                    if decision:
                        st.markdown("### 🎯 Strategy:")
                        st.markdown(decision['strategy'])

                        for model_name in decision['selected_models']:
                            response = get_response(
                                model_name,
                                question,
                                st.session_state.current_chat_id
                            )


if __name__ == "__main__":
    main()
