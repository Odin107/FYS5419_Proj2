
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import circuit_drawer
from qiskit_aer import Aer
import numpy as np

# Create a Quantum Circuit for a 2-qubit iQFT
qc = QuantumCircuit(2)

# Applying the iQFT steps
# Apply Hadamard to the second qubit
qc.h(1)
# Apply inverse Controlled-R2 gate (Controlled-phase gate with negative phase)
qc.cp(-np.pi/2, 0, 1)
# Apply Hadamard to the first qubit
qc.h(0)

# Display the circuit
print(qc.draw())

style = {
    'backgroundcolor': '#FFFFFF',
    'textcolor': '#000000',
    'gatetextcolor': '#000000',
    'subtextcolor': '#000000',
    'linecolor': '#000000',
    'creglinecolor': '#778899',
    'control': 'filled',
    'fontsize': 13,
    'subfontsize': 8,
    'figwidth': 8,
    'figheight': 5,
    'dpi': 100
}

# Save the circuit as a PNG file
circuit_drawer(qc, output='mpl', filename='2_qubit_iQFT.png', style=style)

# Setup the backend
state_simulator = Aer.get_backend('statevector_simulator')

# Transpile the circuit for the backend
transpiled_qc = transpile(qc, state_simulator)

# Run the transpiled circuit on the backend
job = state_simulator.run(transpiled_qc)
result = job.result()

# Get the statevector
statevector = result.get_statevector()

# Output results
print("Statevector:", statevector)


# A function generating the iQFT circuit for n qubits
def iqft(n):
    """
    Apply the Inverse Quantum Fourier Transform (IQFT) on n qubits.

    Args:
        n (int): The number of qubits.

    Returns:
        QuantumCircuit: The quantum circuit representing the IQFT.
    """
    qc = QuantumCircuit(n)
    
    # Apply the inverse QFT gates
    for j in range(n):
        for k in range(j):
            qc.cp(-np.pi / float(2 ** (j - k)), k, j)
        qc.h(j)
    return qc

n = 3  # Change this to the number of qubits you want
qc = iqft(n)
print(qc.draw())

circuit_drawer(qc, output='mpl', filename='n_qubit_iQFT.png', style=style)
