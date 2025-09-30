import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert import HTMLExporter
from huggingface_hub import InferenceClient
from e2b_code_interpreter import Sandbox
from transformers import AutoTokenizer
from traitlets.config import Config
from jupyter_handler import JupyterNotebook
import json


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_and_execute_jupyter_code_cell",
            "description": "A Python code execution environment that runs code in a Jupyter notebook interface. This is stateful - variables and imports persist between executions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide the final answer to the user's question after completing all necessary analysis and computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete final answer to the user's question"
                    },
                },
                "required": ["answer"]
            }
        }
    }
]

TOOLS = TOOLS[:1]

MAX_TURNS = 40


def execute_code(sbx, code):
    execution = sbx.run_code(code, on_stdout=lambda data: print('stdout:', data))
    output = ""
    if len(execution.logs.stdout) > 0:
        output += "\n".join(execution.logs.stdout)
    if len(execution.logs.stderr) > 0:
        output += "\n".join(execution.logs.stderr)
    if execution.error is not None:
        output += execution.error.traceback
    return output, execution


def parse_exec_result_llm(execution, max_code_output=1000):
    output = []

    def truncate_if_needed(text):
        if len(text) > max_code_output:
            return (text[:max_code_output] + f"\n[Output is truncated as it is more than {max_code_output} characters]")
        return text

    if execution.results:
        output.append(truncate_if_needed("\n".join([result.text for result in execution.results])))
    if execution.logs.stdout:
        output.append(truncate_if_needed("\n".join(execution.logs.stdout)))
    if execution.logs.stderr:
        output.append(truncate_if_needed("\n".join(execution.logs.stderr)))
    if execution.error is not None:
        output.append(truncate_if_needed(execution.error.traceback))
    return "\n".join(output)

def clean_messages_for_api(messages):
    """
    Create a clean copy of messages without raw_execution fields for API calls.
    This prevents 413 errors caused by large execution data.
    """
    cleaned_messages = []
    for message in messages:
        cleaned_message = message.copy()
        if "raw_execution" in cleaned_message:
            cleaned_message.pop("raw_execution")
        cleaned_messages.append(cleaned_message)
    return cleaned_messages


def run_interactive_notebook(client, model, messages, sbx, max_new_tokens=512):
    notebook = JupyterNotebook(messages)
    sbx_info = sbx.get_info()
    notebook.add_sandbox_countdown(sbx_info.started_at, sbx_info.end_at)
    yield notebook.render(mode="generating"), notebook.data, messages
    
    max_code_output = 1000
    turns = 0
    done = False

    while not done and (turns <= MAX_TURNS):
        turns += 1
        try:
            # Inference client call - might fail
            response = client.chat.completions.create(
                messages=clean_messages_for_api(messages),
                model=model,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            # Handle inference client errors
            notebook.add_error(f"Inference failed: {str(e)}")
            return notebook.render(), notebook.data, messages

        # Get the response content and tool calls
        full_response = response.choices[0].message.content or ""
        tool_calls = response.choices[0].message.tool_calls or []

        # Add markdown cell for assistant's thinking
        notebook.add_markdown(full_response, "assistant")

        # Handle tool calls
        for tool_call in tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                }
            )

            if tool_call.function.name == "add_and_execute_jupyter_code_cell":
                tool_args = json.loads(tool_call.function.arguments)
            
            notebook.add_code(tool_args["code"])
            yield notebook.render(mode="executing"), notebook.data, messages

            try:
                # Execution sandbox call - might timeout
                execution = sbx.run_code(tool_args["code"])
                notebook.append_execution(execution)
                
            except Exception as e:
                # Handle sandbox timeout/execution errors
                notebook.add_error(f"Code execution failed: {str(e)}")
                return notebook.render(), notebook.data, messages

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": parse_exec_result_llm(execution, max_code_output=max_code_output),
                    "raw_execution": notebook.parse_exec_result_nb(execution)
                }
            )

        if not tool_calls:
            if len(full_response.strip())==0:
                notebook.add_error(f"No tool call and empty assistant response:\n{response.model_dump_json(indent=2)}")
            messages.append({"role": "assistant", "content": full_response})
            done = True
            
        if done:
            yield notebook.render(mode="done"), notebook.data, messages
        else:
            yield notebook.render(mode="generating"), notebook.data, messages