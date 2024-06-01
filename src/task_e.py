import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
from qiskit.circuit.library import QFT, UnitaryGate

def phase_estimation(unitary, num_auxiliary_qubits, shots=8192):
    num_target_qubits = 1  # Assuming a single target qubit for simplicity
    
    # Create the quantum circuit
    qc = QuantumCircuit(num_auxiliary_qubits + num_target_qubits, num_auxiliary_qubits)
    
    # Initialize the target qubit in the |1> state
    qc.x(num_auxiliary_qubits)
    
    # Step 1: Apply Hadamard gates to the auxiliary qubits
    for q in range(num_auxiliary_qubits):
        qc.h(q)
    
    # Step 2: Apply controlled-U gates
    for i in range(num_auxiliary_qubits):
        exponent = 2 ** i
        cu_gate = unitary.power(exponent).control()
        qc.append(cu_gate, [i] + [num_auxiliary_qubits])
    
    # Step 3: Apply inverse QFT to the auxiliary qubits
    qc.append(QFT(num_auxiliary_qubits, do_swaps=False).inverse(), range(num_auxiliary_qubits))
    
    # Step 4: Measure the auxiliary qubits
    qc.measure(range(num_auxiliary_qubits), range(num_auxiliary_qubits))
    
    # Transpile and run the circuit
    simulator = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, backend=simulator, shots=shots)
    result = job.result()
    
    # Get the measurement results
    counts = result.get_counts(qc)
    
    return counts

def estimate_phase(counts, num_auxiliary_qubits):
    total_counts = sum(counts.values())
    weighted_sum = sum(int(state, 2) * count for state, count in counts.items())
    binary_phase = weighted_sum / total_counts / (2 ** num_auxiliary_qubits)
    return binary_phase

# Example unitary (e.g., a phase shift)
theta = np.pi / 4  # Phase
U = UnitaryGate([[1, 0], [0, np.exp(1j * theta)]], label="U")

# Perform phase estimation for one and two auxiliary qubits
counts_1q = phase_estimation(U, 1)
counts_2q = phase_estimation(U, 2)

# Estimate the phases
estimated_phase_1q = estimate_phase(counts_1q, 1)
estimated_phase_2q = estimate_phase(counts_2q, 2)

print("Measurement results for one-qubit phase estimation:", counts_1q)
print("Estimated phase for one-qubit phase estimation:", estimated_phase_1q)

print("Measurement results for two-qubit phase estimation:", counts_2q)
print("Estimated phase for two-qubit phase estimation:", estimated_phase_2q)

# Example usage: Test the QFT for a range of qubits and estimate phases
for n in range(1, 6):  # Testing from 1 to 5 auxiliary qubits
    counts = phase_estimation(U, n)
    estimated_phase = estimate_phase(counts, n)
    print(f"\nMeasurement results for phase estimation with {n} auxiliary qubits:")
    print(counts)
    print(f"Estimated phase for {n} auxiliary qubits:", estimated_phase)

# Plot histograms for visual inspection
for n in range(1, 6):  # Testing from 1 to 5 auxiliary qubits
    counts = phase_estimation(U, n)
    plot_histogram(counts).savefig(f"phase_estimation_{n}_qubits.png")
    estimated_phase = estimate_phase(counts, n)
    print(f"Histogram saved as phase_estimation_{n}_qubits.png")
    print(f"Measurement results for phase estimation with {n} auxiliary qubits:")
    print(counts)
    print(f"Estimated phase for {n} auxiliary qubits:", estimated_phase)