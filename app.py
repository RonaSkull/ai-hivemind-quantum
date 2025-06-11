import gradio as gr
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to speed up startup
import matplotlib.pyplot as plt
import io
import base64
import os
from qiskit.qasm2 import dumps

def run_quantum_demo(circuit_type: str):
    """
    Creates, runs a quantum circuit, and returns the results as a markdown string.
    """
    try:
        # 1. Create Quantum Circuit
        if circuit_type == "Bell State":
            # Creates a 2-qubit Bell state
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
        elif circuit_type == "GHZ State":
            # Creates a 3-qubit GHZ state
            qc = QuantumCircuit(3, 3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)
            qc.measure([0, 1, 2], [0, 1, 2])
        elif circuit_type == "Teleportation":
            # Quantum Teleportation for 1 qubit
            qc = QuantumCircuit(3, 3)
            qc.h(1)
            qc.cx(1, 2)
            qc.cx(0, 1)
            qc.h(0)
            qc.measure([0, 1], [0, 1])
            qc.cx(1, 2)
            qc.cz(0, 2)
            qc.measure([2], [2])
        else:
            return "## Error: Invalid circuit type selected."

        # 2. Simulate the circuit
        simulator = AerSimulator()
        job = simulator.run(qc, shots=1024)
        result = job.result()
        counts = result.get_counts(qc)

        # 3. Visualize the results
        # Generate plot
        fig = plot_histogram(counts, title=f'{circuit_type} - Measurement Results')
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        
        # Encode plot to base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig) # Close the plot to free memory

        # 4. Format the output
        # QASM code
        qasm_code = dumps(qc)
        
        # Measurement results
        counts_str = "\n".join([f"- {state}: {count}" for state, count in counts.items()])

        # Create markdown output
        output_md = f"""
        ## Quantum Circuit: {circuit_type}

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

    except Exception as e:
        return f"## An unexpected error occurred: {str(e)}"

# Gradio Interface
demo = gr.Interface(
    fn=run_quantum_demo,
    inputs=gr.Dropdown(
        ["Bell State", "GHZ State", "Teleportation"], 
        label="Select Quantum Circuit",
        value="Bell State"
    ),
    outputs=gr.Markdown(label="Results"),
    title="Interactive Quantum Circuit Demo",
    description="Select a quantum circuit to run on a simulator. The QASM code, measurement results, and a histogram will be displayed.",
    allow_flagging="never"
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get('PORT', 7860)))
