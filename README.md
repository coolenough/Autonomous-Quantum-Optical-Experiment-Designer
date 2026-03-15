# 🌌 Autonomous Quantum Optical Experiment Designer
### **Neuro-Symbolic Agent for Photonic State Preparation**
**Physis Techne Symposium '26 | Machine Learning Challenge**

## 🎯 Overview
This project implements an **Artificial Intelligence system** that autonomously designs quantum optical experiment setups. Given a target quantum state $\rho_{output}$ defined on an $n$-qubit photonic Hilbert space, the model generates an experimental configuration capable of preparing that state[cite: 9]. The system focuses on encoding information using **polarization and spatial modes**[cite: 11].

## 🛠️ Component Library
The library consists of experimental apparatus commonly used in photonic quantum research:
* **SPDC Sources:** Spontaneous Parametric Down-Conversion sources producing the Bell state $|\phi^{+}\rangle$.
* **Half-Wave Plates (HWP):** Used for precise polarization manipulation.
* **50:50 Beam Splitters (BS):** Facilitates coherent mixing of spatial modes.
* **Phase Shifters:** Capable of applying arbitrary phase shifts.
* **Detectors:** Includes both threshold and photon-number-resolving detectors.

## 🧠 Algorithm & Architecture
Our solution utilizes a **Neuro-Symbolic** approach to meet the challenge of autonomous design:
* **RAG Node:** Indexes local research papers and books to retrieve specialized theoretical methods.
* **Transformer Node:** Predicts the optimal hardware sequence based on target vector features.
* **PINN Node:** Simulates the hardware setup using **QuTiP** to calculate the fidelity of the output state.
* **Self-Correction:** An autonomous logic loop that re-consults the knowledge base if the calculated fidelity is below the target threshold.

## 📊 Performance Metrics & Constraints
The agent operates within strict physical and resource constraints:
* **Max Qubits:** Designed for up to 4-qubit states.
* **Resource Efficiency:** Limited to a maximum of 3 SPDC sources per experiment.
* **Fidelity Metric:** Evaluated using the trace fidelity formula: $F(\rho_{target},\rho_{out})=(Tr\sqrt{\sqrt{\rho_{target}}\rho_{out}\sqrt{\rho_{target}}})^2$.

## 📂 Reproduction
1.  **Dependencies:** Install all required libraries via `requirements.txt`.
2.  **Knowledge Base:** Place your supplementary research PDFs in the main directory.
3.  **Execution:** Run `python quantum_agent.py` to start the autonomous design process.

---

### 🎥 Video Presentation
Our 5-minute explanation of the algorithm beauty, runtime, and results can be found here: **[Your Video Link]**.
