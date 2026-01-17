# MAPM: A Multi-Agent Framework for the Automatic Generation of Business Process Model

This project is based on a multi-agent framework and automatically generates standard BPMN 2.0 process models (XML) and visual diagrams (SVG) from natural language descriptions.
The synthetic dataset used in this project is located in the **`Complex-BPMN-Bench`** folder in the project's root directory.
## 🛠️ Prerequisites

1.  **Python 3.12+**
2.  **Graphviz** (System-level software, used for drawing):
    *   **Windows**: Download and install the installation package. Note: During installation, please select "Add Graphviz to the system PATH for all users".
    *   **Mac**: `brew install graphviz`
    *   **Linux**: `sudo apt-get install graphviz`

## 📦 Installation steps

1.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Model Preparation:**
    Please ensure that the locally fine-tuned DeBERTa model folder `final_activity_classifier_deberta_PMo` is placed in the project's root directory.

3.  **Configure API Key:**
    Open `config/settings.py` (or modify the environment variables directly) and fill in your information:
    * `MODELSCOPE_API_KEY` (used for LLM calls)
    * `TAVILY_API_KEY` (for search tools)

## 🚀 How to run the program

Run the main program directly:

```bash
python main.py
