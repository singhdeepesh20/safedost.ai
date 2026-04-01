<div align="center">

# SafeDost.AI

### A Digital Companion for Women’s Safety

</div>

---

## Overview

**SafeDost.AI** is an AI-powered safety assistant designed to support women with **real-time guidance, awareness, and reliable safety information**.

Built as a **chat-based intelligent system**, it provides **actionable advice in unsafe situations**, powered by verified knowledge sources and modern AI technologies.

SafeDost.AI is more than an application — it is a **digital companion focused on safety, trust, and empowerment**.

---

## Problem Statement

Women often face safety risks in public spaces, transport, and digital environments. Access to **quick, reliable, and actionable guidance** during such situations is limited.

SafeDost.AI addresses this gap by delivering **instant, context-aware safety assistance**.

---

## Key Features

* 🛡️ **Real-Time Safety Guidance** → Immediate help in unsafe situations
* 📚 **Verified Knowledge Base** → Laws, helplines, and safety manuals
* 🔎 **Smart Retrieval System** → Accurate answers using document-based search
* 🧠 **AI-Powered Responses** → Clear, actionable, and context-aware advice
* 💬 **Chat-Based Interface** → Simple and familiar WhatsApp-style interaction
* 🔗 **Source Transparency** → Shows references for trust and reliability

---

## System Architecture

```
User Query → Chat Interface
        ↓
 Knowledge Retrieval (FAISS)
        ↓
 Context Generation
        ↓
 Groq LLM
        ↓
 Actionable Safety Guidance + Source References
```

---

## How It Works

### 📚 Knowledge Base

* Built from curated documents on women’s safety
* Includes laws, helplines, and awareness resources
* Extendable via `./data` directory

### 🔍 Smart Search (FAISS)

* Converts documents into embeddings
* Retrieves the most relevant safety information instantly

### 🧠 AI Engine

* Powered by Groq LLM for fast and clear responses
* Ensures guidance is understandable and practical

### 💬 Chat Interface

* Designed for ease of use
* Enables quick interaction during critical moments

---

## Tech Stack

* **LLM**: Groq API
* **Framework**: LangChain
* **Embeddings**: HuggingFace
* **Vector Store**: FAISS
* **Frontend**: Streamlit (chat interface)

---

## Real-World Impact

### 👩‍🦰 For Women

* Instant access to safety guidance
* Awareness of rights, laws, and helplines
* Increased confidence in unsafe situations

### 🌍 For Society

* Promotes safety awareness
* Reduces information gaps in emergencies
* Supports digital empowerment

---

## What Makes It Different

* Domain-specific focus on **women’s safety**
* Provides **practical, real-world actions** (not generic advice)
* Transparent responses with **source references**
* Built for **speed, reliability, and accessibility**

---

## Future Scope

* Voice-based emergency interaction
* Location-based safety alerts
* Integration with emergency services
* Multilingual support for wider accessibility

---

## Vision

SafeDost.AI aims to ensure that:

> Every woman feels supported, informed, and empowered — anytime, anywhere.

It envisions a world where **technology acts as a constant companion for safety**, bridging the gap between risk and response.

---

<div align="center">

### "In a world full of risks, every woman deserves a dost."

</div>

