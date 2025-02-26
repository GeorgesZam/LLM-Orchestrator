# 🤖 LLM Orchestrator

**LLM Orchestrator** est une application open source qui permet d'interagir intelligemment avec plusieurs modèles de langage (LLM) via une interface web conviviale développée avec **Streamlit**. Elle sélectionne automatiquement le modèle le plus adapté à votre question, améliore itérativement les réponses et gère les conversations de manière fluide. Que vous ayez besoin d'aide pour du code, des analyses ou des questions générales, **LLM Orchestrator** vous offre une expérience utilisateur optimale.

---

## 🚀 Fonctionnalités principales

- **Sélection intelligente des modèles** : Choisit le modèle LLM le plus pertinent en fonction de votre question (code, analyse, général, etc.).
- **Amélioration itérative des réponses** : Affine les réponses en utilisant plusieurs modèles pour garantir la meilleure qualité possible.
- **Gestion des conversations** : Sauvegarde l'historique des chats et maintient le contexte pour des échanges cohérents.
- **Adaptabilité aux ressources machine** : Prend en compte la RAM et la mémoire GPU pour recommander des modèles compatibles.
- **Support multilingue** : Détecte la langue de la question et ajuste les réponses en conséquence (bonus pour le français avec `mistral`).
- **Interface intuitive** : Interface web simple et efficace grâce à **Streamlit**.

---

## 🛠️ Technologies utilisées

- **[Streamlit](https://streamlit.io/)** : Pour l'interface utilisateur interactive.
- **[Ollama](https://ollama.ai/)** : Pour l'intégration et la gestion des modèles LLM.
- **[NetworkX](https://networkx.org/)** et **[Pyvis](https://pyvis.readthedocs.io/)** : Pour la visualisation des graphes (si applicable).
- **[Pandas](https://pandas.pydata.org/)** : Pour la manipulation des données.

---

## 📦 Installation et configuration

### Prérequis
- **Python 3.8+** installé.
- **Ollama** installé et configuré sur votre machine. [Instructions d'installation d'Ollama](https://ollama.ai/docs/installation).
- Les modèles LLM nécessaires doivent être téléchargés via Ollama (par exemple, `ollama pull llama2`).

### Étapes d'installation
1. Clonez le repository :
   ```bash
   git clone https://github.com/votre-utilisateur/llm-orchestrator.git
   cd llm-orchestrator
