# Hybrid Invoice Intelligence 🤖🏢
### Autonomous Enterprise Finance Agent (Kaggle Agents Capstone)



## 📖 Project Overview
**Hybrid Invoice Intelligence** is an autonomous AI agent designed to automate the Accounts Payable (AP) workflow for enterprise finance teams. It solves the problem of manual data entry and compliance checks by combining:
1.  **Cloud Vision (Google Gemini 2.0 Flash):** For high-precision OCR and extracting structured data from complex invoice images.
2.  **Local Reasoning (Gemma 3:1B via Ollama):** For privacy-preserving expense categorization, anomaly detection, and email drafting.

This hybrid architecture ensures sensitive business logic runs locally while leveraging state-of-the-art cloud vision for data ingestion.

## 🏗️ Architecture
The system is orchestrated using **LangGraph** with a cyclic state machine:
* **👁️ Extractor Node:** Multimodal extraction using Gemini Vision.
* **🧠 Validator Node:** Local Gemma model categorizes expenses and checks for anomalies (e.g., High Value > ₹50k, Weekend Invoices).
* **💱 FX Engine:** Real-time currency conversion to a base reporting currency.
* **⚖️ Evaluator Node:** Quality gate to flag low-confidence extractions for human review.
* **💾 Reporter Node:** Writes to a persistent SQLite ledger and drafts notification emails.

## ⚡ Key Features
* **Hybrid AI Stack:** Seamless handoff between Cloud (Vision) and Local (Reasoning) models.
* **Cyber-Glass UI:** Modern, dark-mode dashboard built with Streamlit.
* **Dynamic Policies:** Configurable thresholds for anomaly detection (no hardcoded rules).
* **Audit Trail:** Full transparency with color-coded logs and a persistent history ledger.
* **Self-Healing DB:** Automatic schema migration for database updates.

## 🛠️ Installation & Run

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed and running (`ollama serve`)
* Google Gemini API Key

### 1. Clone & Install
```bash
git clone [https://github.com/YOUR_USERNAME/invoice-agent-capstone.git](https://github.com/YOUR_USERNAME/invoice-agent-capstone.git)
cd invoice-agent-capstone
pip install -r requirements.txt

### 2. The YouTube Link for Demonstrations
https://youtu.be/Qu_b_TptuVA  