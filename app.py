import os
import gradio as gr
from gradio.utils import get_space
from huggingface_hub import InferenceClient
from e2b_code_interpreter import Sandbox
from pathlib import Path
from transformers import AutoTokenizer
import json
from openai import OpenAI
from huggingface_hub import HfApi, HfFolder
from jupyter_handler import JupyterNotebook

# Optional .env loading when not running in a HF space
if not get_space():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except (ImportError, ModuleNotFoundError):
        pass

import json as _json
try:
    import eval_tool
except ImportError:
    eval_tool = None

from utils import run_interactive_notebook,clean_messages_for_api,parse_exec_result_llm

E2B_API_KEY = os.environ.get("E2B_API_KEY", "e2b_test_key")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
DEFAULT_MAX_TOKENS = 512
SANDBOXES = {}
SANDBOX_TIMEOUT = 420
TMP_DIR = './tmp/'
model="Qwen3-Coder-480B-A35B-Instruct"  # Using a known working model
init_notebook = JupyterNotebook()

if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

with open(TMP_DIR+"jupyter-agent.ipynb", 'w', encoding='utf-8') as f:
    json.dump(JupyterNotebook().data, f, indent=2)

with open("ds-system-prompt.txt", "r") as f:
    DEFAULT_SYSTEM_PROMPT = f.read()
DEFAULT_SYSTEM_PROMPT = """You are a coding agent with access to a Jupyter Kernel. \
When possible break down tasks step-by-step. \
The following files are available (if any):
{}

List of available packages:
# Jupyter server requirements
jupyter-server==2.16.0
ipykernel==6.29.5
ipython==9.2.0

orjson==3.10.18
pandas==2.2.3
matplotlib==3.10.3
pillow==11.3.0

# Latest version for
e2b_charts

# Other packages
aiohttp==3.12.14
beautifulsoup4==4.13.4
bokeh==3.7.3
gensim==4.3.3 # unmaintained, blocking numpy and scipy bump
imageio==2.37.0
joblib==1.5.0
librosa==0.11.0
nltk==3.9.1
numpy==1.26.4 # bump blocked by gensim
numba==0.61.2
opencv-python==4.11.0.86
openpyxl==3.1.5
plotly==6.0.1
kaleido==1.0.0
pytest==8.3.5
python-docx==1.1.2
pytz==2025.2
requests==2.32.4
scikit-image==0.25.2
scikit-learn==1.6.1
scipy==1.13.1 # bump blocked by gensim
seaborn==0.13.2
soundfile==0.13.1
spacy==3.8.2 # doesn't work on 3.13.x
textblob==0.19.0
tornado==6.5.1
urllib3==2.5.0
xarray==2025.4.0
xlrd==2.0.1
sympy==1.14.0

If you need to install additional packages:
1. install uv first with `pip install uv` 
2. then use uv to install the package with `uv pip install PACKAGE_NAME --system`.

"""

def execute_jupyter_agent(
    user_input, files, message_history, request: gr.Request
):

    if request.session_hash not in SANDBOXES:
        SANDBOXES[request.session_hash] = Sandbox.create(api_key=E2B_API_KEY, timeout=SANDBOX_TIMEOUT)
    sbx = SANDBOXES[request.session_hash]

    save_dir = os.path.join(TMP_DIR, request.session_hash)
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, 'jupyter-agent.ipynb')

    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(init_notebook.data, f, indent=2)
    yield init_notebook.render(), message_history, save_dir

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
    
    # Debug: Print first few characters of API key to verify it's set
    print(f"Using API key: {HF_TOKEN[:10]}...{HF_TOKEN[-5:]}" if HF_TOKEN else "No API key set")

    filenames = []
    if files is not None:
        for filepath in files:
            filpath = Path(filepath)
            with open(filepath, "rb") as file:
                print(f"uploading {filepath}...")
                sbx.files.write(filpath.name, file)
                filenames.append(filpath.name)

    sytem_prompt = DEFAULT_SYSTEM_PROMPT
    # Initialize message_history if it doesn't exist
    if len(message_history) == 0:
        if files is None:
            sytem_prompt = sytem_prompt.format("- None")
        else:
            sytem_prompt = sytem_prompt.format("- " + "\n- ".join(filenames))

        message_history.append(
            {
                "role": "system",
                "content": sytem_prompt,
            }
        )
    message_history.append({"role": "user", "content": user_input})

    #print("history:", message_history)

    for notebook_html, notebook_data, messages in run_interactive_notebook(
        client, model, message_history, sbx,
    ):
        message_history = messages
        
        yield notebook_html, message_history, TMP_DIR+"jupyter-agent.ipynb"
    
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=2)
    yield notebook_html, message_history, save_dir

def clear(msg_state, request: gr.Request):
    if request.session_hash in SANDBOXES:
        SANDBOXES[request.session_hash].kill()
        SANDBOXES.pop(request.session_hash)

    msg_state = []
    return init_notebook.render(), msg_state


css = """
#component-0 {
    height: 100vh;
    overflow-y: auto;
    padding: 20px;
}

.gradio-container {
    height: 100vh !important;
}

.contain {
    height: 100vh !important;
}
"""


# Create the interface
with gr.Blocks() as demo:
    msg_state = gr.State(value=[])

    html_output = gr.HTML(value=JupyterNotebook().render())
    
    user_input = gr.Textbox(
        #value="Write code to multiply three numbers: 10048, 32, 19", lines=3, label="User input"
        value="Solve the Lotka-Volterra equation and plot the results. Do it step by step and explain what you are doing and in the end make a super nice and clean plot.", label="Agent task"
    )
    
    with gr.Row():
        generate_btn = gr.Button("Run!")
        clear_btn = gr.Button("Clear Notebook")
    
    with gr.Accordion("Upload files ⬆ | Download notebook⬇", open=False):
        files = gr.File(label="Upload files to use", file_count="multiple")
        file = gr.File(TMP_DIR+"jupyter-agent.ipynb", label="Download Jupyter Notebook")

    powered_html = gr.HTML("""\
        <p align="center">
             <img style="max-height:100px; max-width:100%; height:auto;"src="https://huggingface.co/spaces/lvwerra/jupyter-agent-2/resolve/main/powered-by.png" alt="Powered by" />
        </p>""")
    

    generate_btn.click(
        fn=execute_jupyter_agent,
        inputs=[user_input, files, msg_state],
        outputs=[html_output, msg_state, file],
        show_progress="hidden",
    )

    clear_btn.click(fn=clear, inputs=[msg_state], outputs=[html_output, msg_state])

    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js=""" () => {
    if (document.querySelectorAll('.dark').length) {
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    }
}
"""
    )
    # ---- Evaluation wrapper for run_interactive_notebook (added at $PLACEHOLDER$) ----

    # Save the original run_interactive_notebook before wrapping
    _orig_run_interactive_notebook = run_interactive_notebook

    def _load_eval_fn():
        if eval_tool is None:
            return lambda html: {"error": "eval_tool not available"}

        for cand in ["evaluate_session"]:
            if hasattr(eval_tool, cand) and callable(getattr(eval_tool, cand)):
                return getattr(eval_tool, cand)
        return lambda html: {"error": "No suitable function (evaluate_session/evaluate/score/run) found in eval_tool.py"}

    _eval_fn = _load_eval_fn()

    def run_interactive_notebook(*args, **kwargs):
        """
        Wrapper around original run_interactive_notebook that:
        - Yields original streaming outputs unchanged while the agent is generating.
        - After the original generator is exhausted, runs eval_tool on the final notebook_html.
        - Appends the scores (pretty JSON) to the notebook_html and yields one extra final update.
        """
        if _orig_run_interactive_notebook is None:
            raise RuntimeError("Original run_interactive_notebook not found to wrap.")

        final_tuple = None
        for _tpl in _orig_run_interactive_notebook(*args, **kwargs):
            final_tuple = _tpl  # (_html, _data, _msgs)
            yield _tpl  # stream through unchanged

        if final_tuple is None:
            return  # Nothing was generated

        notebook_html, notebook_data, messages = final_tuple

        # Run evaluation ONLY after generation is complete
        try:
            if _eval_fn.__name__ == 'evaluate_session':
                # evaluate_session expects (notebook_html, message_history, save_dir)
                result = _eval_fn(notebook_html, messages, None)
                scores = result.to_dict() if hasattr(result, 'to_dict') else result
            else:
                # Fallback for other evaluation functions
                scores = _eval_fn(notebook_html)
        except Exception as e:
            scores = {"error": f"Evaluation failed: {e}"}

        # Append scores to HTML (simple formatting)
        try:
            scores_pretty = _json.dumps(scores, indent=2, ensure_ascii=False)
        except Exception:
            scores_pretty = str(scores)

        appended_html = (
            notebook_html
            + "<hr><div style='font-family:monospace'>"
            + "<h3>Evaluation Scores</h3><pre>"
            + scores_pretty
            + "</pre></div>"
        )

        # Yield one extra (final) tuple with scores appended
        yield appended_html, notebook_data, messages
demo.launch(ssr_mode=False)
