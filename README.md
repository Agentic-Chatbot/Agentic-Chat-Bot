# Agentic Chatbot for Robotics Club 🤖

## 📌 Project Overview

The **Agentic Chatbot** is an intelligent assistant designed for the **Robotics Club**.  
It aims to streamline club activities by providing technical support, document retrieval, task management, event assistance, and data-driven evaluation.

The chatbot leverages **agentic workflows, context graphs, and Google service integrations** to act as a reliable and efficient digital coordinator.

---

## 🚀 Features

- **GPT-powered Q&A** – Answer technical queries using LangChain + OpenAI API
- **Document Retrieval (RAG)** – Search Google Drive documents with ChromaDB + embeddings
- **Contextual Knowledge Graphs** – Neo4j integration for member ↔ project mapping
- **Agentic Planning** – Multi-step workflows via LangGraph
- **Email Notifications** – Automated updates using Gmail API
- **Google Forms Integration** – Share and track event registrations
- **Task Tracking & Skill Evaluation** – Neo4j skill mapping, logs, and analytics with Pandas
- **Event Conduct & Judging** – Automated evaluation using workflow scripts and metrics

---

## 🛠️ Tech Stack

- **Core AI:** LangChain, OpenAI API
- **Document Search:** ChromaDB, embeddings, chunking
- **Graphs & Context:** Neo4j
- **Agent Workflows:** LangGraph
- **Google Services:** Gmail API, Google Drive API, Google Forms API
- **Backend Framework:** FastMCP (for MCP server implementation)
- **Utilities:** PyMuPDF / pdfplumber, dotenv, google-auth
- **Deployment:** GitHub, Render / GCP

---

## 📅 Project Timeline

| Phase                              | Dates                   | Duration | Key Tasks                                                                                                                                                  |
| ---------------------------------- | ----------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1 – Learning & Setup**     | 10 Aug – 7 Sept         | 29 days  | Learn LangChain, LangGraph, RAG, Neo4j basics, Google APIs; set up Git/GitHub & FastMCP skeleton                                                           |
| **Phase 2 – MVP Build**            | 8 Sept – 19 Sept        | 11 days  | Basic Q&A (LangChain + OpenAI), Google Drive API, ChromaDB RAG, minimal testing                                                                            |
| **Phase 3 – Feature Expansion**    | 27 Sept – 17 Oct        | 30 days  | Sprint 1: FastMCP server, Neo4j, Gmail API, Task tracking & skill evaluation. <br> Sprint 2: Google Forms, LangGraph workflows, Event conduct & evaluation |
| **Phase 4 – Optimization**         | November                | 10 days  | Bug fixes, optimization, prompt tuning, API stability                                                                                                      |
| **Phase 5 – Final Testing & Docs** | 1st week after end-sems | 7 days   | End-to-end testing, documentation, deployment                                                                                                              |

---

## 📂 Repository Structure

```
AGENTIC CHATBOT/
│
├── Task 1 - Project Proposal/
│ └── Agentic_Chatbot_Proposal.pdf/
│
├── Task 2 - Chatbot/
│ └── agentic_task_planner.py/
│ └── README.md/
│ └── requirements.txt/
│
├── Task 3 - LSTM Model/
│ └── bayesian_trials.csv/
│ └── LSTM_Bayesian_Optimization.ipynb/
│ └── LSTM_Random_Search.ipynb/
│ └── random_search_results.csv/
│ └── readme.md/
│ └── requirements.txt/
│
└── README.md
```

---

## ✅ Deliverables

- Fully functional **Agentic Chatbot** for Robotics Club
- Seamless **Google services integration**
- **Task & skill evaluation** with analytics
- Automated **event management & judging system**
- Comprehensive **documentation & deployment** guide

---
