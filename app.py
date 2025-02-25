import streamlit as st
import ollama
import json
import time
import re
import psutil
import platform
import subprocess

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

# Initialisation des variables de session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0
if 'best_response' not in st.session_state:
    st.session_state.best_response = None
if 'client' not in st.session_state:
    st.session_state.client = ollama.Client(host="http://localhost:11434")
if 'stop_generation' not in st.session_state:
    st.session_state.stop_generation = False


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
        available_models = {}  # Dictionnaire pour stocker {nom_base: nom_complet}

        # Extraire les noms de modèles avec leurs tags
        for model_obj in response.models:
            if hasattr(model_obj, 'model'):
                full_name = str(model_obj.model)  # Convertir en string
                base_name = full_name.split(':')[0]  # ex: "qwen2"
                if base_name in MODEL_INFO:  # Vérifier si le modèle est configuré
                    available_models[base_name] = full_name

        return available_models

    except Exception as e:
        st.error(f"Erreur lors de la récupération des modèles : {e}")
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


def get_response(model, question):
    """Obtient la réponse d'un modèle avec streaming"""
    try:
        response_container = st.empty()
        full_response = ""

        # Boutons de contrôle
        skip = st.button("⏭️ Skip", key=f"skip_{model}")
        delire = st.button("🚫 Délire", key=f"delire_{model}")

        if skip or delire:
            status = 'skipped' if skip else 'delire'
            message = "*Réponse ignorée*" if skip else "*❌ Modèle marqué comme délirant*"
            response_container.markdown(message)
            return {'status': status, 'content': None, 'model': model}

        stream_response = st.session_state.client.chat(
            model=model,
            messages=[{'role': 'user', 'content': question}],
            stream=True
        )

        for chunk in stream_response:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                full_response += content
                response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)
        return {'status': 'success', 'content': full_response, 'model': model}

    except Exception as e:
        st.error(f"Erreur avec {model}: {e}")
        return {'status': 'error', 'content': None, 'model': model}


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


def get_manager_prompt(question, max_time, available_models, MODEL_INFO):
    """Crée le prompt pour le LLM manager"""
    return f"""Vous êtes un assistant qui choisit le meilleur modèle selon la question.

Question : {question}
Temps disponible : {max_time} secondes
Modèles disponibles : {', '.join(available_models.keys())}

Règles de sélection :
1. Questions scientifiques/complexes → mistral ou phi4
2. Questions de programmation → llama2
3. Questions simples/reformulation → qwen2

Analysez la question et choisissez le modèle le plus adapté.
Répondez en JSON :
{{
    "selected_models": ["nom_du_modele"],
    "reformulated_question": "question claire",
    "strategy": "raison du choix"
}}"""


def get_manager_decision(question, max_time):
    available_models = get_available_models()

    if not available_models:
        st.error("Aucun modèle disponible")
        return None

    # Utiliser qwen2 comme manager
    manager_model = available_models.get('qwen2', list(available_models.values())[0])

    # Détection du type de question
    is_scientific = any(word in question.lower() for word in ['pourquoi', 'comment', 'origine', 'science', 'théorie'])
    is_code = any(word in question.lower() for word in ['code', 'programme', 'python', 'javascript'])

    try:
        response = st.session_state.client.chat(
            model=manager_model,
            messages=[{
                'role': 'user',
                'content': get_manager_prompt(question, max_time, available_models, MODEL_INFO)
            }],
            stream=False
        )

        try:
            content = response['message']['content'].strip()
            if content.startswith("```"):
                content = re.sub(r'^```.*\n|```$', '', content)

            decision = json.loads(content)

            # Forcer le choix du modèle selon le type de question
            if is_scientific and 'mistral' in available_models:
                selected_model = 'mistral'
            elif is_code and 'llama2' in available_models:
                selected_model = 'llama2'
            else:
                selected_model = decision['selected_models'][0]

            if selected_model in available_models:
                decision['selected_models'] = [available_models[selected_model]]
                return decision
            else:
                st.error(f"Modèle {selected_model} non disponible")
                return None

        except Exception as e:
            st.error(f"Erreur de parsing: {str(e)}")
            return None

    except Exception as e:
        st.error(f"Erreur du manager: {str(e)}")
        return None


# Interface principale
st.markdown("<h1 class='main-header'>🤖 LLM Manager Pro</h1>", unsafe_allow_html=True)

# Afficher la configuration machine
show_machine_config()

# Paramètres
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Modèles disponibles")
    available_models = get_available_models()  # Récupérer les modèles réellement disponibles

    if not available_models:
        st.warning("⚠️ Serveur Ollama non accessible ou aucun modèle installé")
    else:
        for model_name, full_name in available_models.items():
            if model_name in MODEL_INFO:  # Vérifier si on a les infos pour ce modèle
                info = MODEL_INFO[model_name]
                with st.expander(f"{full_name} ({info['size']})"):
                    st.write("Benchmarks:")
                    for benchmark, score in info["benchmarks"].items():
                        st.write(f"- {benchmark}: {score}")
                    st.write("Spécialités:", ", ".join(info["specialties"]))
            else:
                # Pour les modèles installés mais non configurés dans MODEL_INFO
                with st.expander(f"{full_name}"):
                    st.write("*Informations non disponibles pour ce modèle*")

with col2:
    question = st.text_area("Posez votre question:")
    max_time = st.slider(
        "Temps maximum (secondes)",
        min_value=30,
        max_value=180,
        value=60,
        key="max_time_slider"
    )

    if st.button("Obtenir la meilleure réponse"):
        if question:
            with st.spinner("🤔 Analyse de la question..."):
                decision = get_manager_decision(question, max_time)

            if decision:
                st.markdown("### 🎯 Stratégie du manager:")
                st.markdown(decision['strategy'])

                st.markdown("### 📝 Question reformulée:")
                st.markdown(decision['reformulated_question'])

                # Créer un conteneur pour les réponses
                responses_container = st.container()

                with responses_container:
                    # Créer les colonnes pour les réponses
                    num_models = len(decision['selected_models'])
                    if num_models > 0:
                        cols = st.columns(min(num_models, 2))  # Maximum 2 colonnes
                        responses = []

                        for i, (model_name, col) in enumerate(zip(decision['selected_models'], cols)):
                            with col:
                                st.markdown(f"#### Réponse de {model_name}:")
                                response = get_response(model_name, decision['reformulated_question'])
                                responses.append(response)

                        # Synthèse si nécessaire
                        valid_responses = [r for r in responses if r['status'] == 'success']
                        if len(valid_responses) > 1:
                            st.markdown("### 💡 Synthèse finale:")
                            synthesis = synthesize_responses(responses)
                            st.markdown(synthesis)
            else:
                st.error("Veuillez installer au moins un des modèles configurés sur le serveur Ollama.")
        else:
            st.warning("Veuillez entrer une question.")
