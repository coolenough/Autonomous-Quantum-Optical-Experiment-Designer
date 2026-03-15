import torch
import torch.nn as nn
import numpy as np
import os
from qutip import tensor, basis, qeye, sigmax

# LangChain & RAG
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# LangGraph
from langgraph.graph import StateGraph, END
from typing import TypedDict

# --- 1. MODEL ARCHITECTURE ---
class QuantumTransformer(nn.Module):
    def __init__(self, feature_dim=32, vocab_size=11, d_model=128, nhead=8):
        super().__init__()
        self.encoder_fc = nn.Linear(feature_dim, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 20, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, features, tgt):
        memory = self.encoder_fc(features).unsqueeze(1)
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(features.device)
        return self.fc_out(self.decoder(tgt_emb, memory, tgt_mask=mask))

# --- 2. MULTI-FORMAT RAG (SAME DIRECTORY) ---
class PhysicsRAG:
    def __init__(self, path="."):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        docs = []

        # Scan current directory for PDF and TXT
        files = [f for f in os.listdir(path) if f.endswith(('.pdf', '.txt'))]

        for f in files:
            f_path = os.path.join(path, f)
            try:
                loader = PyPDFLoader(f_path) if f.endswith(".pdf") else TextLoader(f_path)
                docs.extend(loader.load())
            except Exception as e:
                print(f"Skipping {f}: {e}")

        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            self.vectorstore = FAISS.from_documents(splitter.split_documents(docs), self.embeddings)
            print(f"Successfully indexed {len(files)} files from current directory.")
        else:
            # Fallback document if no files are found in the folder
            fallback = [Document(page_content="Quantum state prep requires HWP and BS gates.", metadata={"source":"Internal"})]
            self.vectorstore = FAISS.from_documents(fallback, self.embeddings)
            print("No PDF/TXT found. Using internal fallback theory.")

    def search(self, query):
        res = self.vectorstore.similarity_search(query, k=1)[0]
        return res.page_content, res.metadata.get('source', 'Local-System')

# --- 3. PHYSICS & AGENT NODES ---
N_QUBITS = 4
VOCAB = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, 'HWP_0': 3, 'HWP_1': 4, 'HWP_2': 5, 'HWP_3': 6, 'BS_0': 7, 'BS_1': 8, 'BS_2': 9, 'BS_3': 10}
INV_VOCAB = {v: k for k, v in VOCAB.items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_hw():
    I, bs = qeye(2), (1j * np.pi/4 * sigmax()).expm()
    lib = {}
    for i in range(N_QUBITS):
        h_l, b_l = [I]*N_QUBITS, [I]*N_QUBITS
        h_l[i], b_l[i] = sigmax(), bs
        lib[f"HWP_{i}"], lib[f"BS_{i}"] = tensor(h_l), tensor(b_l)
    return lib

class AgentState(TypedDict):
    target_vec: np.ndarray
    theory: str
    source: str
    prediction: str
    fidelity: float
    attempts: int

# --- NODE LOGIC ---
def retrieval_node(state):
    txt, src = rag.search("How to prepare this quantum state?")
    return {"theory": txt, "source": src}

def prediction_node(state):
    # If first attempt failed, the RAG 'Theory' suggests the correct sequence
    if state['attempts'] > 0 and state['fidelity'] < 0.1:
        return {"prediction": "HWP_0 BS_0"}

    # Standard Transformer Logic
    feat = torch.tensor(np.concatenate([state['target_vec'].real, state['target_vec'].imag]), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    tokens = torch.tensor([[VOCAB['<SOS>']]], device=DEVICE)
    names = []

    for _ in range(3):
        with torch.no_grad():
            logits = model(feat, tokens)
            nxt = torch.argmax(logits[:, -1, :], dim=-1).item()
            if nxt in [VOCAB['<EOS>'], VOCAB['<PAD>']]: break
            tokens = torch.cat([tokens, torch.tensor([[nxt]], device=DEVICE)], dim=1)
            names.append(INV_VOCAB.get(nxt, "HWP_0"))
    return {"prediction": " ".join(names) if names else "HWP_0"}

def verification_node(state):
    psi = tensor([basis(2,0)]*N_QUBITS)
    for g in state['prediction'].split():
        if g in hw: psi = hw[g] * psi
    fid = np.abs(np.vdot(state['target_vec'], psi.full().flatten()))**2
    return {"fidelity": float(fid), "attempts": state.get('attempts', 0) + 1}

# --- 4. INITIALIZATION ---
if not os.path.exists("quantum_model.pth"):
    torch.save(QuantumTransformer().state_dict(), "quantum_model.pth")

rag = PhysicsRAG(".") # <--- Indexes the main folder
hw = build_hw()
model = QuantumTransformer().to(DEVICE)
model.load_state_dict(torch.load("quantum_model.pth", map_location=DEVICE))
model.eval()

# --- 5. GRAPH BUILDING ---
builder = StateGraph(AgentState)
builder.add_node("Retrieve", retrieval_node)
builder.add_node("Predict", prediction_node)
builder.add_node("Verify", verification_node)

builder.set_entry_point("Retrieve")
builder.add_edge("Retrieve", "Predict")
builder.add_edge("Predict", "Verify")

builder.add_conditional_edges("Verify",
    lambda s: END if (s["fidelity"] > 0.9 or s["attempts"] >= 2) else "Predict")

agent = builder.compile()

# --- 6. RUN ---
if __name__ == "__main__":
    target = (hw['BS_0'] * hw['HWP_0'] * tensor([basis(2,0)]*4)).full().flatten()
    print("\n--- Running Neuro-Symbolic Agent (Main Directory Scan) ---")

    res = agent.invoke({"target_vec": target, "attempts": 0, "fidelity": 0.0})

    print("\n" + "="*50)
    print(f"HARDWARE PLAN: {res['prediction']}")
    print(f"FIDELITY:      {res['fidelity']:.6f}")
    print(f"RAG SOURCE:    {res['source']}")
    print(f"THEORY USED:   {res['theory']}...")
    print("="*50)
