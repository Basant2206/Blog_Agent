# 🧠 Blog Generation System (LangGraph + Groq)

A fully agent-based **blog generation system** built using **LangGraph**, **LangChain**, and the **Groq LLM**, capable of researching and writing high-quality blogs automatically.

---

## 🚀 Features

- **Agent based workflow** powered by **LangGraph**
- **Uses Groq Chat Model** (`mixtral-8x7b` or `llama3-8b`)
- **Free Research Tools:** Wikipedia + DuckDuckGo (no API key needed)
- **Automatic blog structure:**
  - Heading
  - Introduction
  - Content
  - Summary
- **No UI** — runs via CLI and prints the full blog

---

## 🧩 Architecture

The system runs as a **LangGraph workflow** with the following nodes:

1. **Research Node** → Gathers info from Wikipedia & DuckDuckGo  
2. **Outline Node** → Creates a blog outline using the Groq model  
3. **Content Node** → Expands the outline into a full blog  
4. **Finalize Node** → Combines everything into the final blog text  

Flow:  
`Research ➜ Outline ➜ Content ➜ Finalize ➜ END`

---

## 🛠️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Basant2206/Blog_Agent.git
cd Blog_Agent



# To Create a virtual environment
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows


# To Install dependencies
pip install -r requirements.txt


#Create a .env file in the project root
Then in .env
GROQ_API_KEY="your_groq_api_key_here"


# To run in terminal type
python Agent.py
