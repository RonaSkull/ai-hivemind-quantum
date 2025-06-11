import gradio as gr
from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps as qasm3_dumps
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib
matplotlib.use('Agg') # Configure Matplotlib for server environment
import matplotlib.pyplot as plt
import io
import base64
import os
import requests

# --- Hugging Face Inference API Configuration ---
# This will now work on the Render platform
API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder2-3b"
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY") # Get API key from environment variables

def format_results(qc, counts, title):
    """Helper function to format the circuit results into a markdown string."""
    try:
        qasm_code = qasm3_dumps(qc)
    except Exception as e:
        qasm_code = f"Could not generate QASM code. Error: {e}"

    fig = plot_histogram(counts, title=f'{title} - Measurement Results')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # Sort counts for consistent display
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    counts_str = "\n".join([f"- {state}: {count}" for state, count in sorted_counts])

    output_md = f"""
    ## Quantum Results: {title}
    ### QASM Code
    ```qasm
    {qasm_code}
    ```
    ### Measurement Results (1024 shots)
    {counts_str}
    ### Histogram
    ![Histogram](data:image/png;base64,{img_str})
    """
    return output_md

def run_predefined_circuit(circuit_type: str):
    """Creates and runs a predefined quantum circuit."""
    try:
        if circuit_type == "Bell State":
            qc = QuantumCircuit(2, 2); qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
        elif circuit_type == "GHZ State":
            qc = QuantumCircuit(3, 3); qc.h(0); qc.cx(0, 1); qc.cx(0, 2); qc.measure([0, 1, 2], [0, 1, 2])
        elif circuit_type == "Teleportation":
            qc = QuantumCircuit(3, 3); qc.h(1); qc.cx(1, 2); qc.barrier(); qc.cx(0, 1); qc.h(0)
            qc.measure([0, 1], [0, 1]); qc.barrier(); qc.cx(1, 2); qc.cz(0, 2); qc.measure([2], [2])
        else:
            return "## Error: Invalid circuit type selected."

        simulator = AerSimulator()
        job = simulator.run(qc, shots=1024)
        counts = job.result().get_counts(qc)
        return format_results(qc, counts, circuit_type)
    except Exception as e:
        return f"## An unexpected error occurred: {str(e)}"

def generate_and_run_code(prompt: str):
    """Generates Qiskit code from a prompt, executes it, and returns the results."""
    yield "## Generating... Contacting the AI HiveMind on Render. This may take up to 60 seconds."

    if not HF_API_KEY:
        yield "## FATAL ERROR: `HUGGINGFACE_API_KEY` not found. Please ensure it is set correctly in the Render environment variables."
        return
    if not prompt:
        yield "## Please enter a prompt."
        return

    full_prompt = f"""Create a Python function `run_generated_circuit()` that uses Qiskit to build and run a quantum circuit for the following request: "{prompt}".
The function must not take any arguments.
The function must return two values: the QuantumCircuit object and the final counts dictionary.
The function must contain all necessary imports inside it.

Here is the complete Python code:
```python
"""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": 512, "return_full_text": False}}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            yield f"## Error from API: {response.text}"
            return
        
        generated_code = response.json()[0]['generated_text']
        
        if "```" in generated_code:
            generated_code = generated_code.split("```")[0]
            
    except requests.RequestException as e:
        yield f"## API Request Failed: {e}"
        return
    except (KeyError, IndexError) as e:
        yield f"## API Error: Could not parse the response: {response.text} (Error: {e})"
        return

    try:
        local_scope = {}
        exec(generated_code, globals(), local_scope)
        run_circuit_func = local_scope.get('run_generated_circuit')
        
        if not callable(run_circuit_func):
            yield f"## Execution Error: The generated code did not define `run_generated_circuit` function correctly.\n\n**Model Output:**\n```python\n{generated_code}\n```"
            return
            
        qc, counts = run_circuit_func()
        yield format_results(qc, counts, f"AI-Generated: {prompt}")

    except Exception as e:
        yield f"## An error occurred while executing the generated code: {str(e)}\n\n**Generated Code:**\n```python\n{generated_code}\n```"

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI HiveMind Quantum Demonstrator")
    
    with gr.Tabs():
        with gr.TabItem("AI-Generated Circuit"):
            gr.Markdown("Give the AI a prompt to generate a quantum circuit. Try financial models, molecular simulations, or famous algorithms!")
            with gr.Row():
                prompt_input = gr.Textbox(label="Enter your prompt", placeholder="e.g., Implement Grover's algorithm for 2 qubits to find '11'", scale=4)
                submit_ai = gr.Button("Generate & Run", variant="primary", scale=1)
            with gr.Column():
                output_ai = gr.Markdown(label="Results")
        
        with gr.TabItem("Predefined Circuits"):
            gr.Markdown("Select a foundational quantum circuit to understand the building blocks of quantum computing.")
            with gr.Row():
                circuit_input = gr.Dropdown(
                    ["Bell State", "GHZ State", "Teleportation"], 
                    label="Select Quantum Circuit",
                    value="Bell State",
                    scale=4
                )
                submit_predefined = gr.Button("Run Circuit", variant="primary", scale=1)
            with gr.Column():
                output_predefined = gr.Markdown(label="Results")

    submit_ai.click(fn=generate_and_run_code, inputs=prompt_input, outputs=output_ai)
    submit_predefined.click(fn=run_predefined_circuit, inputs=circuit_input, outputs=output_predefined)

if __name__ == "__main__":
    demo.launch()
