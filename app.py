import streamlit as st
import requests
import json
import re
import psutil
import platform
import subprocess
import base64
import time
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from functools import lru_cache
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_orchestrator")

# Basic configuration
st.set_page_config(page_title="🤖 LLM Orchestrator", layout="wide")

# Custom CSS styles
st.markdown("""
<style>
    .response-container {
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
    }
    .sidebar-info {
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .model-benchmark {
        padding: 8px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .status-ok {
        color: green;
        font-weight: bold;
    }
    .status-warning {
        color: orange;
        font-weight: bold;
    }
    .status-error {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class SystemResources:
    """Data class to store system hardware information"""
    ram_gb: float
    gpu_gb: float
    max_model_size_b: int


class SystemAnalyzer:
    """Analyzes hardware capabilities to determine compatible model sizes"""

    def __init__(self):
        """Initialize the system analyzer and gather system information"""
        self.resources = self._analyze_system()
        logger.info(f"System resources: RAM={self.resources.ram_gb}GB, GPU={self.resources.gpu_gb}GB, "
                    f"Max model size={self.resources.max_model_size_b}B")

    def _analyze_system(self) -> SystemResources:
        """Analyze system resources and return a SystemResources object"""
        ram_gb = self._get_ram()
        gpu_gb = self._get_gpu()
        max_model_size = self._calc_max_model_size(ram_gb, gpu_gb)

        return SystemResources(
            ram_gb=ram_gb,
            gpu_gb=gpu_gb,
            max_model_size_b=max_model_size
        )

    @staticmethod
    def _get_ram() -> float:
        """Get the total system RAM in GB"""
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)

    @staticmethod
    def _get_gpu() -> float:
        """Get the GPU memory in GB, if available"""
        try:
            if platform.system() == "Darwin":
                return 4.0  # Metal for macOS (approximate)

            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
            ).decode()
            return round(int(output.strip()) / 1024, 1)
        except Exception as e:
            logger.warning(f"Could not detect GPU: {e}")
            return 0.0

    @staticmethod
    def _calc_max_model_size(ram_gb: float, gpu_gb: float) -> int:
        """Calculate maximum model size in billions of parameters"""
        if gpu_gb >= 24: return 70
        if gpu_gb >= 12: return 30
        if gpu_gb >= 6: return 14
        return 7 if ram_gb >= 16 else 3

    @property
    def ram(self) -> float:
        """Get the total system RAM in GB"""
        return self.resources.ram_gb

    @property
    def gpu(self) -> float:
        """Get the GPU memory in GB"""
        return self.resources.gpu_gb

    @property
    def max_model_size(self) -> int:
        """Get the maximum model size in billions of parameters"""
        return self.resources.max_model_size_b


@dataclass
class ModelPerformance:
    """Data class to store model benchmark data for a specific category"""
    general: int = 70
    math: int = 70
    code: int = 70
    speed: int = 70
    web_qa: int = 70
    vision: int = 0  # 0 means not capable


class ModelBenchmark:
    """Provides benchmark data for models across different task categories"""

    def __init__(self, benchmark_file: str = 'model_benchmarks.json'):
        """Initialize the benchmark manager with data from file or defaults"""
        self.benchmark_file = benchmark_file
        self.benchmark_data = self._load_benchmarks()
        self.save_benchmark_data()  # Save benchmark data after loading
        logger.info(f"Loaded benchmark data for {len(self.benchmark_data)} models")

    def _load_benchmarks(self) -> Dict[str, Dict[str, int]]:
        """Load benchmark data from file or use defaults"""
        try:
            # Try to load from a local file
            with open(self.benchmark_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load benchmark file: {e}. Using default benchmarks.")
            # If file doesn't exist, return default benchmarks
            return {
                # General models
                "mistral:7b": {"general": 80, "math": 75, "code": 73, "speed": 85, "web_qa": 77},
                "llama2:7b": {"general": 75, "math": 70, "code": 68, "speed": 80, "web_qa": 72},
                "llama3:8b": {"general": 85, "math": 80, "code": 83, "speed": 78, "web_qa": 84},
                "phi3:3b": {"general": 68, "math": 65, "code": 67, "speed": 95, "web_qa": 64},
                "phi3:7b": {"general": 82, "math": 78, "code": 80, "speed": 82, "web_qa": 79},
                "phi3:14b": {"general": 88, "math": 85, "code": 86, "speed": 70, "web_qa": 85},
                "deepseek:7b": {"general": 78, "math": 82, "code": 81, "speed": 78, "web_qa": 75},
                "deepseek:14b": {"general": 85, "math": 90, "code": 89, "speed": 65, "web_qa": 82},
                "mixtral:8x7b": {"general": 90, "math": 88, "code": 87, "speed": 60, "web_qa": 89},
                "solar:10.7b": {"general": 84, "math": 82, "code": 80, "speed": 72, "web_qa": 83},
                "llama3:70b": {"general": 95, "math": 93, "code": 94, "speed": 30, "web_qa": 94},
                "gemma:7b": {"general": 78, "math": 74, "code": 75, "speed": 83, "web_qa": 74},
                "gemma:2b": {"general": 65, "math": 60, "code": 62, "speed": 90, "web_qa": 60},

                # Vision models
                "llava:7b": {"general": 75, "math": 70, "code": 68, "speed": 75, "web_qa": 72, "vision": 85},
                "llava:13b": {"general": 82, "math": 78, "code": 76, "speed": 65, "web_qa": 80, "vision": 90},
                "bakllava:7b": {"general": 78, "math": 72, "code": 69, "speed": 73, "web_qa": 75, "vision": 88},

                # Code specific models
                "codellama:7b": {"general": 72, "math": 73, "code": 85, "speed": 80, "web_qa": 70},
                "codellama:13b": {"general": 78, "math": 80, "code": 92, "speed": 65, "web_qa": 75},
                "wizardcoder:7b": {"general": 70, "math": 72, "code": 88, "speed": 78, "web_qa": 68},

                # Math specialized models
                "tinyllama:1.1b": {"general": 55, "math": 50, "code": 48, "speed": 100, "web_qa": 45},

                # Add generic entries for common model patterns
                "default:7b": {"general": 75, "math": 72, "code": 70, "speed": 80, "web_qa": 70},
                "default:13b": {"general": 82, "math": 80, "code": 78, "speed": 65, "web_qa": 78},
                "default:30b": {"general": 88, "math": 85, "code": 84, "speed": 45, "web_qa": 86},
                "default:70b": {"general": 94, "math": 92, "code": 90, "speed": 30, "web_qa": 93},
            }

    def get_benchmark(self, model_name: str, category: str = "general") -> int:
        """
        Get benchmark score for a model in a specific category

        Args:
            model_name: Name of the model (e.g., "mistral:7b")
            category: Category to get benchmark for (e.g., "general", "math", "code")

        Returns:
            Benchmark score (0-100)
        """
        # Try to get exact match
        if model_name in self.benchmark_data:
            return self.benchmark_data[model_name].get(category, 70)

        # Try to match by prefix (e.g., llama2:7b-chat -> llama2:7b)
        for prefix in self.benchmark_data:
            base_model = prefix.split(':')[0]
            if model_name.startswith(base_model):
                size_match = re.search(r'(\d+)b', prefix)
                if size_match and size_match.group(1) in model_name:
                    return self.benchmark_data[prefix].get(category, 70)

        # Fall back to generic size-based benchmarks
        size_match = re.search(r'(\d+)b', model_name)
        if size_match:
            size = size_match.group(1)
            default_key = f"default:{size}b"
            if default_key in self.benchmark_data:
                return self.benchmark_data[default_key].get(category, 70)

        # Default fallback
        return 70  # Average score if no match found

    def get_model_performance(self, model_name: str) -> ModelPerformance:
        """Get complete performance profile for a model"""
        performance = ModelPerformance()

        # Fill in all categories
        for category in vars(performance):
            setattr(performance, category, self.get_benchmark(model_name, category))

        return performance

    def save_benchmark_data(self) -> None:
        """Save benchmark data to a JSON file"""
        with open(self.benchmark_file, 'w') as f:
            json.dump(self.benchmark_data, f)


@dataclass
class OllamaStatus:
    """Data class to store Ollama service status"""
    running: bool
    message: str
    version: str = "unknown"


@dataclass
class GenerationRequest:
    """Data class to store parameters for a generation request"""
    model: str
    prompt: str
    images: List[bytes] = None
    stream: bool = True
    temperature: float = 0.7
    max_tokens: int = 2048


class OllamaInterface:
    """Interface for Ollama with error handling and mock responses"""

    def __init__(self, base_url: str = "http://localhost:11434", use_mock: bool = False):
        """
        Initialize the Ollama interface

        Args:
            base_url: URL of the Ollama API
            use_mock: Whether to use mock responses (for when Ollama isn't available)
        """
        self.base_url = base_url
        self.use_mock = use_mock
        self.status = self._check_status()
        self.models = self._fetch_models()

        if self.status.running:
            logger.info(f"Connected to Ollama {self.status.version} with {len(self.models)} models")
        else:
            logger.warning(f"Ollama not available: {self.status.message}. Using mock mode.")

    def _check_status(self) -> OllamaStatus:
        """Check if Ollama is running properly"""
        if self.use_mock:
            return OllamaStatus(running=False, message="Using mock mode (Ollama not detected)")

        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=2)
            if response.status_code == 200:
                version = response.json().get("version", "unknown")
                return OllamaStatus(running=True, message=f"Connected", version=version)
            return OllamaStatus(running=False, message=f"Error: Status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            return OllamaStatus(running=False, message="Error: Cannot connect to Ollama")
        except Exception as e:
            return OllamaStatus(running=False, message=f"Error: {str(e)}")

    def _fetch_models(self) -> List[str]:
        """Fetch available models from Ollama"""
        if self.use_mock:
            return ["mistral:7b", "llama2:7b", "phi3:3b", "llava:7b", "tinyllama:1.1b"]

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
            return ["mistral:7b"]  # Default fallback
        except Exception as e:
            logger.warning(f"Error fetching models: {e}")
            # Fall back to mock models if Ollama isn't running
            self.use_mock = True
            return ["mistral:7b", "llama2:7b", "phi3:3b", "llava:7b", "tinyllama:1.1b"]

    def generate(self, request: GenerationRequest) -> str:
        """
        Generate a response from the model

        Args:
            request: Generation request parameters

        Returns:
            Generated text response
        """
        if self.use_mock:
            return self._mock_response(request.prompt, request.model)

        # Construct the request payload
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
            "temperature": request.temperature,
            "num_predict": request.max_tokens
        }

        # Add images if provided (for multimodal models)
        if request.images:
            payload["images"] = [base64.b64encode(img).decode('utf-8') for img in request.images]

        try:
            # Create a placeholder for streaming responses
            if request.stream:
                response_placeholder = st.empty()
                full_response = ""

                # Make the streaming request
                with requests.post(f"{self.base_url}/api/generate", json=payload, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    full_response += chunk["response"]
                                    response_placeholder.markdown(f"*Generating...*\n\n{full_response}▌")
                            except json.JSONDecodeError:
                                pass

                response_placeholder.empty()
                return full_response
            else:
                # Non-streaming request
                with st.spinner(f"Generating with {request.model}..."):
                    response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)

                    if response.status_code == 200:
                        result = response.json()
                        return result.get("response", "No response generated")
                    else:
                        st.error(f"Error from Ollama API: {response.status_code}")
                        return f"Error generating response: Status code {response.status_code}"
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            st.error(f"Error connecting to Ollama: {str(e)}")
            return self._mock_response(request.prompt, request.model)

    def _mock_response(self, prompt: str, model: str) -> str:
        """Generate a mock response when Ollama is unavailable"""
        # For math questions, provide simple answers
        if re.search(r'(\d+\s*[\+\-\*\/]\s*\d+\s*=?)', prompt):
            math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=?', prompt)
            if math_match:
                try:
                    a, op, b = math_match.groups()
                    a, b = int(a), int(b)
                    result = 0
                    if op == '+':
                        result = a + b
                    elif op == '-':
                        result = a - b
                    elif op == '*':
                        result = a * b
                    elif op == '/' and b != 0:
                        result = a / b

                    return f"The answer to {a} {op} {b} is {result}."
                except Exception as e:
                    logger.warning(f"Error in mock math response: {e}")

        return f"""[Mock Response from {model}] 

You asked: "{prompt[:50]}..."

This is a simulated response since Ollama is not available or had an error. In a real setup with Ollama running correctly, you would receive an actual AI-generated response here.

To use this application with real LLM capabilities:
1. Install Ollama from https://ollama.ai
2. Run the Ollama service
3. Pull a model like 'ollama pull mistral:7b'
4. Restart this application
"""


@dataclass
class SearchResult:
    """Data class to store web search results"""
    title: str
    url: str
    description: str
    content: str = ""


class WebSearch:
    """Performs web searches without API dependencies"""

    def __init__(self, timeout: int = 10):
        """Initialize the web search with configurable timeout"""
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def search(self, query: str, max_results: int = 2) -> List[SearchResult]:
        """
        Search the web for information

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results
        """
        url = f"https://search.brave.com/search?q={query.replace(' ', '+')}"

        try:
            with st.spinner("Searching the web..."):
                response = requests.get(url, headers=self.headers, timeout=self.timeout)

                if response.status_code != 200:
                    logger.warning(f"Web search failed with status code {response.status_code}")
                    return []

                soup = BeautifulSoup(response.text, "html.parser")
                results = []

                for i, result in enumerate(soup.select(".snippet, .organic-result")):
                    if i >= max_results:
                        break

                    title_elem = result.select_one(".title, .snippet-title")
                    url_elem = result.select_one(".url, .result-header")
                    description_elem = result.select_one(".description, .snippet-description")

                    if title_elem:
                        title = title_elem.text.strip()
                        url = url_elem.text.strip() if url_elem else ""
                        description = description_elem.text.strip() if description_elem else ""

                        # Try to get more content by following the link
                        content = description
                        try:
                            if url and url.startswith(('http://', 'https://')):
                                content_response = requests.get(url, headers=self.headers, timeout=5)
                                if content_response.status_code == 200:
                                    content_soup = BeautifulSoup(content_response.text, 'html.parser')
                                    # Extract paragraphs for better content
                                    paragraphs = content_soup.find_all('p')
                                    if paragraphs:
                                        content = ' '.join([p.text for p in paragraphs[:5]])
                        except Exception as e:
                            logger.warning(f"Error fetching additional content: {e}")

                        results.append(SearchResult(
                            title=title,
                            url=url,
                            description=description,
                            content=content[:500]  # Limit content length
                        ))

                return results

        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            st.warning(f"Error during web search: {str(e)}")
            return []


@dataclass
class ModelSelection:
    """Data class to store model selection results"""
    model: str
    reason: str
    score: int
    alternatives: List[str]


class ModelSelector:
    """Selects appropriate model based on question type, benchmarks and hardware"""

    def __init__(self, system_analyzer: SystemAnalyzer, benchmarks: ModelBenchmark):
        """
        Initialize the model selector

        Args:
            system_analyzer: System analyzer for hardware compatibility checking
            benchmarks: Model benchmarks for performance comparison
        """
        self.system = system_analyzer
        self.benchmarks = benchmarks

    def select_model(self, question: str, available_models: List[str], has_images: bool = False,
                     use_web: bool = False) -> ModelSelection:
        """
        Select the most appropriate model for the given question

        Args:
            question: User's question/prompt
            available_models: List of available models
            has_images: Whether the question includes images
            use_web: Whether web search is enabled

        Returns:
            ModelSelection object with the selected model and alternatives
        """
        # First, filter by hardware compatibility
        compatible_models = self._filter_compatible(available_models)

        if not compatible_models:
            return ModelSelection(
                model=available_models[0] if available_models else "mistral:7b",
                reason="No hardware-compatible models found, using default",
                score=0,
                alternatives=[]
            )

        # If question has images, prioritize vision models
        if has_images:
            vision_models = [m for m in compatible_models if any(v in m.lower()
                                                                 for v in ["llava", "vision", "bakllava", "clip"])]
            if vision_models:
                best_vision = self._rank_models(vision_models, "vision")[0]
                return ModelSelection(
                    model=best_vision[0],
                    reason="Selected best vision model for image analysis",
                    score=best_vision[1],
                    alternatives=[m[0] for m in self._rank_models(vision_models, "vision")[1:3]]
                )

        # Handle simple math questions with a tiny model
        if self._is_simple_math(question):
            small_models = [m for m in compatible_models if any(s in m.lower()
                                                                for s in ["phi", "tiny", "small", "mini"])]
            if small_models:
                best_small = self._rank_models(small_models, "speed")[0]
                return ModelSelection(
                    model=best_small[0],
                    reason="Selected fast model for simple math question",
                    score=best_small[1],
                    alternatives=[m[0] for m in self._rank_models(small_models, "speed")[1:3]]
                )

        # Detect question type for specialized models
        question_type = self._detect_question_type(question)

        # For web searches, prioritize web_qa capability
        if use_web:
            category = "web_qa"
        else:
            category = question_type

        # Get ranked models for the specific category
        ranked_models = self._rank_models(compatible_models, category)

        # Return the best model with alternatives
        return ModelSelection(
            model=ranked_models[0][0],
            reason=f"Selected best model for {category} tasks",
            score=ranked_models[0][1],
            alternatives=[m[0] for m in ranked_models[1:3]]
        )

    def _filter_compatible(self, models: List[str]) -> List[str]:
        """Filter models based on hardware compatibility"""
        result = []
        for model in models:
            # Extract model size if available in the name
            size_match = re.search(r'(\d+)b', model.lower())
            if size_match:
                size = int(size_match.group(1))
                if size <= self.system.max_model_size:
                    result.append(model)
            else:
                # If size not in name, assume it's compatible
                result.append(model)
        return result or models  # Return all models if none are compatible

    @staticmethod
    def _is_simple_math(question: str) -> bool:
        """Check if this is a simple math question"""
        question = question.lower()
        # Look for simple arithmetic patterns
        return bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+', question))

    @staticmethod
    def _detect_question_type(question: str) -> str:
        """Detect the type of question asked"""
        question = question.lower()

        if any(word in question for word in ["code", "program", "function", "bug", "coding"]):
            return "code"
        elif any(word in question for word in ["math", "calculate", "equation", "formula"]):
            return "math"
        else:
            return "general"

    def _rank_models(self, models: List[str], category: str) -> List[Tuple[str, int]]:
        """Rank models by benchmark score for the given category"""
        scores = []
        for model in models:
            score = self.benchmarks.get_benchmark(model, category)
            scores.append((model, score))

        # Sort by score, descending
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def show_model_comparison(self, models: List[str], category: str = "general") -> None:
        """Show a comparison of models for a specific category"""
        if not models:
            st.warning("No models available for comparison")
            return

        st.subheader(f"Model Comparison for {category.capitalize()} Tasks")

        # Get scores for each model
        scores = []
        for model in models:
            scores.append({
                "Model": model,
                "Score": self.benchmarks.get_benchmark(model, category),
                "Size": self._extract_size(model)
            })

        # Sort by score, descending
        scores = sorted(scores, key=lambda x: x["Score"], reverse=True)

        # Display as a table
        st.table(scores)

    @staticmethod
    def _extract_size(model: str) -> str:
        """Extract model size from name"""
        size_match = re.search(r'(\d+)b', model.lower())
        return f"{size_match.group(1)}B" if size_match else "Unknown"


@dataclass
class ConversationEntry:
    """Data class to store a conversation entry"""
    question: str
    response: str
    model: str
    time: float
    images: int = 0
    web_results: int = 0


class ConversationHistory:
    """Manages the conversation history"""

    def __init__(self, max_entries: int = 20, history_file: str = 'conversation_history.json'):
        """
        Initialize the conversation history

        Args:
            max_entries: Maximum number of entries to store
            history_file: File to save conversation history
        """
        self.max_entries = max_entries
        self.entries: List[ConversationEntry] = []
        self.history_file = history_file
        self.load_history()  # Charger l'historique existant

    def load_history(self) -> None:
        """Load conversation history from a JSON file"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.entries = [ConversationEntry(**entry) for entry in json.load(f)]

    def save_history(self) -> None:
        """Save conversation history to a JSON file"""
        with open(self.history_file, 'w') as f:
            json.dump([entry.__dict__ for entry in self.entries], f)

    def add_entry(self, entry: ConversationEntry) -> None:
        """Add an entry to the conversation history"""
        self.entries.append(entry)
        self.save_history()  # Enregistrer après ajout
        # Trim history if needed
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
            self.save_history()  # Enregistrer après suppression

    def get_recent_entries(self, count: int = 5) -> List[ConversationEntry]:
        """Get the most recent entries"""
        return self.entries[-count:] if self.entries else []

    def clear(self) -> None:
        """Clear the conversation history"""
        self.entries = []


class LLMOrchestrator:
    """Main application class that ties together all components"""

    def __init__(self):
        """Initialize the LLM Orchestrator application"""
        # Initialize session state if needed
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = ConversationHistory()

        # Initialize system components
        self.system = SystemAnalyzer()
        self.benchmarks = ModelBenchmark()

        # Check if Ollama is available
        self.ollama_available = self._check_ollama_availability()
        self.ollama = OllamaInterface(use_mock=not self.ollama_available)

        # Initialize other components
        self.web_search = WebSearch()
        self.selector = ModelSelector(self.system, self.benchmarks)

    @staticmethod
    @lru_cache(maxsize=1)  # Cache the result to avoid repeated checks
    def _check_ollama_availability() -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def render_sidebar(self) -> None:
        """Render the sidebar with system information and available models"""
        with st.sidebar:
            st.title("🖥️ System Info")

            # System info
            st.markdown(
                f"<div class='sidebar-info'>RAM: {self.system.ram} GB<br>GPU: {self.system.gpu} GB<br>Max model size: {self.system.max_model_size}B</div>",
                unsafe_allow_html=True)

            # Ollama status
            status_class = "status-ok" if self.ollama.status.running else "status-error"
            st.markdown(
                f"<div class='sidebar-info'>Ollama: <span class='{status_class}'>{self.ollama.status.message}</span></div>",
                unsafe_allow_html=True)

            # Available models
            st.title("🤖 Available Models")
            if self.ollama.models:
                for model in self.ollama.models:
                    # Get performance data
                    perf = self.benchmarks.get_model_performance(model)

                    # Skip models that are too large for hardware
                    size_match = re.search(r'(\d+)b', model.lower())
                    if size_match and int(size_match.group(1)) > self.system.max_model_size:
                        st.markdown(f"<div class='model-benchmark' style='opacity: 0.5;'>{model} (incompatible)</div>",
                                    unsafe_allow_html=True)
                        continue

                    # Show model with performance indicators
                    st.markdown(
                        f"""<div class='model-benchmark'>
                            <b>{model}</b><br>
                            General: {perf.general}/100 | Math: {perf.math}/100<br>
                            Code: {perf.code}/100 | Speed: {perf.speed}/100
                            {f"<br>Vision: {perf.vision}/100" if perf.vision > 0 else ""}
                        </div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No models available. Please install Ollama and pull models.")

            # Instructions/Help
            with st.expander("ℹ️ Help & Instructions"):
                st.markdown("""
                **Getting Started**:
                1. Install [Ollama](https://ollama.ai)
                2. Pull models: `ollama pull mistral:7b`
                3. Ask questions in the main panel

                **Tips**:
                - The app will automatically select the best model for your question
                - Enable Web Search for up-to-date information
                - For image analysis, use models with vision capabilities
                """)

            # Clear conversation history
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history.clear()
                st.success("Conversation history cleared!")

    def render_main_interface(self) -> None:
        """Render the main interface for user interaction"""
        st.title("🤖 LLM Orchestrator")
        st.markdown("Ask questions and get answers from locally running language models")

        # Input area
        question = st.text_area("Ask a question:", height=100)

        # Advanced options in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            enable_web = st.checkbox("Enable Web Search", value=False,
                                     help="Search the web for up-to-date information")

        with col2:
            uploaded_images = st.file_uploader("Upload Images (for vision models)",
                                               accept_multiple_files=True, type=["jpg", "jpeg", "png"])
            has_images = uploaded_images is not None and len(uploaded_images) > 0

        with col3:
            manual_model = st.selectbox("Override Model Selection",
                                        ["Auto Select"] + self.ollama.models,
                                        help="Manually select a model instead of automatic selection")
            use_auto_select = manual_model == "Auto Select"

        # Generation parameters
        with st.expander("Generation Parameters"):
            temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                                    help="Higher values make output more random, lower values more deterministic")
            max_tokens = st.slider("Max Tokens", min_value=256, max_value=4096, value=2048, step=256,
                                   help="Maximum number of tokens to generate")

        # Process the query when the user clicks the button
        if st.button("Generate Response") and question:
            self._process_query(
                question=question,
                enable_web=enable_web,
                uploaded_images=uploaded_images,
                use_auto_select=use_auto_select,
                manual_model=manual_model if not use_auto_select else None,
                temperature=temperature,
                max_tokens=max_tokens
            )

        # Display conversation history
        st.subheader("📝 Conversation History")
        history = st.session_state.conversation_history.get_recent_entries()

        if not history:
            st.info("No conversation history yet. Ask a question to get started!")

        # Display history in reverse order (newest first)
        for entry in reversed(history):
            with st.expander(f"Q: {entry.question[:50]}..." if len(entry.question) > 50 else f"Q: {entry.question}"):
                st.markdown(f"**Using model**: {entry.model}")
                st.markdown(f"**Time**: {entry.time:.2f} seconds")

                # Add badges for web search and images if used
                badges = []
                if entry.web_results > 0:
                    badges.append(f"🌐 {entry.web_results} web results")
                if entry.images > 0:
                    badges.append(f"🖼️ {entry.images} images")

                if badges:
                    st.markdown(" | ".join(badges))

                st.markdown("**Response**:")
                st.markdown(f"<div class='response-container'>{entry.response}</div>", unsafe_allow_html=True)

    def _process_query(self, question: str, enable_web: bool, uploaded_images: List,
                       use_auto_select: bool, manual_model: str = None, temperature: float = 0.7,
                       max_tokens: int = 2048) -> None:
        """Process a user query and generate a response"""
        start_time = time.time()
        web_results = []
        images = []

        # Convert uploaded images to bytes
        if uploaded_images:
            for img in uploaded_images:
                images.append(img.read())

        # Search the web if enabled
        if enable_web:
            web_results = self.web_search.search(question)

            # Format web results as context
            if web_results:
                web_context = "I found the following information that might help:\n\n"
                for i, result in enumerate(web_results, 1):
                    web_context += f"{i}. {result.title}\n"
                    web_context += f"   URL: {result.url}\n"
                    web_context += f"   {result.content}\n\n"

                # Prepend web results to question
                question_with_context = f"{web_context}\n\nBased on the above information and your knowledge, please answer: {question}"
            else:
                question_with_context = question
        else:
            question_with_context = question

        # Select model
        if use_auto_select:
            selection = self.selector.select_model(
                question=question,
                available_models=self.ollama.models,
                has_images=bool(images),
                use_web=enable_web
            )
            model = selection.model

            # Show model selection explanation
            st.info(f"Selected model: **{model}** - {selection.reason}")
        else:
            model = manual_model

        # Create generation request
        request = GenerationRequest(
            model=model,
            prompt=question_with_context,
            images=images if images else None,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Generate response
        with st.spinner(f"Generating response with {model}..."):
            response = self.ollama.generate(request)

        # Calculate time
        elapsed_time = time.time() - start_time

        # Add to conversation history
        entry = ConversationEntry(
            question=question,
            response=response,
            model=model,
            time=elapsed_time,
            images=len(images) if images else 0,
            web_results=len(web_results)
        )
        st.session_state.conversation_history.add_entry(entry)

        # Display response
        st.markdown("### Response:")
        st.markdown(f"<div class='response-container'>{response}</div>", unsafe_allow_html=True)
        st.markdown(f"Generated with **{model}** in {elapsed_time:.2f} seconds")


def main():
    """Main entry point of the application"""
    # Initialize and run the orchestrator
    orchestrator = LLMOrchestrator()

    # Render the app components
    orchestrator.render_sidebar()
    orchestrator.render_main_interface()


if __name__ == "__main__":
    main()
