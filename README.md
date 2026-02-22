# MedMap AI 🏥
**Intelligent Medical Consultation & Prescription Management System**

MedMap AI is a full-stack Flask application that uses local LLMs to assist healthcare providers in diagnosing symptoms, extracting data from medical documents, and generating verified prescriptions.



---

## ✨ Key Features

* **AI Symptom Analysis**: Uses **Ollama (Llama 3.2)** to perform medical triage and categorize patient conditions.
* **OCR Document Processing**: Extracts structured medicine data (Generic name, Strength, Form) from PDFs and Images using **PyTesseract**.
* **Smart Prescription Filling**: Cross-references symptoms with a local database to suggest appropriate medications.
* **Safety Guardrails**: AI-powered verification ensures suggested medicines are safe for the specific symptoms provided.
* **Automated Patient Reports**: Generates concise instructions, advantages, and disadvantages for every prescription.
* **RAG (Retrieval-Augmented Generation)**: Uses a training database to provide context-aware medical answers based on uploaded knowledge files.

---

## 🛠️ Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Backend** | Python / Flask |
| **AI/LLM** | Ollama (Llama 3.2:latest) |
| **Database** | SQLite3 |
| **OCR** | Tesseract OCR / PyTesseract |
| **PDF Parsing** | PyPDF2 |
| **Image Processing** | Pillow (PIL) |

---

## 🚀 Installation & Setup

### 1. Prerequisites
* **Python 3.9+**
* **Ollama**: [Download Ollama](https://ollama.com/) and run `ollama pull llama3.2`
* **Tesseract OCR**: 
    * **Windows**: [Install binaries](https://github.com/UB-Mannheim/tesseract/wiki) and add to Environment Path.
    * **Linux**: `sudo apt install tesseract-ocr`
    * **Mac**: `brew install tesseract`

### 2. Clone and Install
```bash
# Clone the repository
git clone [https://github.com/your-username/medmap-ai.git](https://github.com/your-username/medmap-ai.git)
cd medmap-ai

# Install dependencies
pip install flask requests PyPDF2 pytesseract Pillow
