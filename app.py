import gradio as gr
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os

# Qiskit imports for Portfolio Optimization
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.applications import PortfolioOptimization
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.primitives import Sampler

# Qiskit imports for Circuit Simulation
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# --- Portfolio Optimization Functions ---

def get_stock_data(tickers, start_date="2023-01-01", end_date="2024-01-01"):
    """Fetches historical stock data from Yahoo Finance."""
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def run_portfolio_optimization(ticker_string: str):
    """
    Performs quantum portfolio optimization for a given list of stock tickers.
    """
    try:
        # 1. Get Stock Data
        tickers = [ticker.strip().upper() for ticker in ticker_string.split(',')]
        if len(tickers) < 2:
            return "## Error\nPlease provide at least two stock tickers, separated by commas.", None

        stock_data = get_stock_data(tickers)
        if stock_data.empty or stock_data.isnull().values.any():
            return f"## Error\nCould not fetch valid data for all tickers: {tickers}. Please check the symbols.", None

        # 2. Formulate the Optimization Problem
        mu = stock_data.pct_change().mean().values
        sigma = stock_data.pct_change().cov().values
        
        q = 0.5  # Risk aversion factor
        budget = len(tickers) // 2 # Number of assets to select
        
        portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget)
        qp = portfolio.to_quadratic_program()

        # 3. Solve with Quantum Algorithm (QAOA)
        qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), initial_point=[0.0, 0.0])
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        
        optimizer = MinimumEigenOptimizer(qaoa_mes)
        result = optimizer.solve(qubo)
        
        # 4. Format and Return Results
        selection = portfolio.interpret(result)
        allocation = {tickers[i]: val * 100 for i, val in enumerate(selection) if val > 0}

        # Create a plot for visualization
        fig, ax = plt.subplots()
        ax.pie(allocation.values(), labels=allocation.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Optimized Portfolio Allocation')
        
        # Save plot to a buffer
        buf = os.path.join(os.getcwd(), "portfolio.png")
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)

        # Prepare markdown output
        output_md = f"""## Quantum Portfolio Optimization Results

**Optimized Allocation ({budget} assets selected):**
"""
        for stock, pct in allocation.items():
            output_md += f"- **{stock}**: {pct:.2f}%\n"
        
        return output_md, buf

    except Exception as e:
        return f"## An error occurred:\n\n```\n{str(e)}\n```", None

# --- Quantum Circuit Functions ---

def create_bell_circuit():
    """Creates a Bell state circuit."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def create_ghz_circuit():
    """Creates a GHZ state circuit."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc

def create_teleportation_circuit():
    """Creates a quantum teleportation circuit."""
    qc = QuantumCircuit(3, 3)
    qc.rx(np.pi / 4, 0) 
    qc.barrier()
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    qc.barrier()
    qc.cx(1, 2)
    qc.cz(0, 2)
    qc.measure(2, 2)
    return qc

def run_quantum_circuit(circuit_name):
    """Runs a selected quantum circuit and returns results."""
    if circuit_name == "Bell State":
        qc = create_bell_circuit()
    elif circuit_name == "GHZ State":
        qc = create_ghz_circuit()
    else: # Teleportation
        qc = create_teleportation_circuit()

    circuit_diagram_path = os.path.join(os.getcwd(), "circuit.png")
    qc.draw(output='mpl', filename=circuit_diagram_path, style="iqp")

    simulator = AerSimulator()
    job = simulator.run(qc, shots=1024)
    result = job.result()
    counts = result.get_counts(qc)

    histogram_path = os.path.join(os.getcwd(), "histogram.png")
    plot_histogram(counts).savefig(histogram_path)

    counts_str = "### Measurement Results (Counts):\n"
    for outcome, count in counts.items():
        counts_str += f"- `{outcome}`: {count}\n"

    return circuit_diagram_path, counts_str, histogram_path

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🚀 AI-HI Quantum Demonstrations")
    
    with gr.Tabs():
        # Tab 1: Portfolio Optimization
        with gr.TabItem("📈 Quantum Portfolio Optimizer"):
            gr.Markdown("Enter stock tickers (e.g., AAPL, GOOG, MSFT) to find the optimal portfolio allocation using a quantum algorithm (QAOA).")
            
            with gr.Row():
                ticker_input = gr.Textbox(label="Stock Tickers (comma-separated)", placeholder="e.g., AAPL, GOOG, MSFT, NVDA")
                optimizer_btn = gr.Button("Optimize Portfolio")

            with gr.Row():
                optimizer_results = gr.Markdown()
                optimizer_plot = gr.Image(type="filepath")

            optimizer_btn.click(
                fn=run_portfolio_optimization,
                inputs=ticker_input,
                outputs=[optimizer_results, optimizer_plot]
            )

        # Tab 2: Quantum Circuit Demos
        with gr.TabItem("⚛️ Quantum Circuit Simulator"):
            gr.Markdown("Select a fundamental quantum circuit to simulate its execution and view the results.")
            
            with gr.Row():
                circuit_dropdown = gr.Dropdown(
                    ["Bell State", "GHZ State", "Teleportation"], label="Select Circuit"
                )
                circuit_btn = gr.Button("Run Simulation")
            
            with gr.Row():
                circuit_diagram = gr.Image(label="Circuit Diagram")
                with gr.Column():
                    circuit_results = gr.Markdown()
                    circuit_histogram = gr.Image(label="Result Histogram")
            
            circuit_btn.click(
                fn=run_quantum_circuit,
                inputs=circuit_dropdown,
                outputs=[circuit_diagram, circuit_results, circuit_histogram]
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
