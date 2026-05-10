# SHL Conversational Assessment Recommender

This repository contains the solution for the AI Intern Take-home Assignment. The project implements a Conversational Agent using FastAPI and Google Gemini that guides a user from a vague intent to a grounded shortlist of SHL assessments.

## Features
- **Stateless Chat API**: Fully stateless `POST /chat` endpoint that processes the entire conversation history per turn.
- **Intent Recognition**: Intelligently categorizes user queries into Clarify, Search, Compare, or Off-Topic.
- **Robust Schema Validation**: Employs Pydantic models and LLM Structured Outputs to guarantee the exact response schema expected by the automated evaluator.
- **Semantic Retrieval**: Uses a lightweight TF-IDF and Cosine Similarity mechanism to quickly and accurately search the SHL product catalog.
- **Strict Grounding**: The agent will NEVER recommend an assessment that is not present in the provided catalog data.

## Project Structure
- `main.py`: The FastAPI application exposing `/health` and `/chat` endpoints.
- `agent.py`: Orchestrates the LLM logic, including intent extraction, structured generation, and fallback mechanisms.
- `retriever.py`: Loads the raw `shl_product_catalog.json` data and provides TF-IDF vector similarity search.
- `approach.md`: A detailed two-page write-up explaining design choices, retrieval setup, and evaluation strategy.

## Getting Started Locally

### Prerequisites
- Python 3.9+
- A Google Gemini API Key

### Setup
1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/jayesh3103/SHL-Assessment.git
   cd SHL-Assessment
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Configure your API key:
   Create a `.env` file in the root directory (or export the variable directly):
   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

4. Run the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Testing
You can interact with the live API documentation by navigating to `http://localhost:8000/docs` in your browser. Alternatively, use standard curl requests to test the `/chat` endpoint.

---

## Approach and Methodology

### Design Choices & Architecture
The system is built on **FastAPI** to provide a lightweight, stateless API endpoint. The orchestration uses a 3-step LLM pipeline utilizing **Google Gemini-1.5-Flash**. The choice of Gemini Flash was driven by its very large context window, fast time-to-first-token, and robust JSON structured output capabilities, which perfectly fits the strict schema requirements and the 30-second timeout limits.

To ensure robustness against non-deterministic conversations:
- The system explicitly separates "Intent/Query Extraction" from "Generation". 
- Instead of using a fragile, multi-step ReAct agent, we use a single-pass extraction to map the stateless history into a concrete set of search parameters, followed by a deterministic retrieval phase, and concluding with a generation phase heavily grounded by the retrieved catalog.

### Retrieval Setup
We employ a lightweight **TF-IDF + Cosine Similarity** implementation using `scikit-learn`. 
- **Preprocessing**: On startup, the JSON catalog is loaded. Invalid control characters are ignored, and items are flattened into search documents composed of `name`, `description`, `job_levels`, and `keys`.
- **Retrieval**: We match user queries against this corpus and retrieve the top 30 most relevant items. Given the catalog size (377 items), 30 items offer a very high Recall@30 while keeping the final context size small enough (~5k tokens) to ensure sub-2-second LLM generation.

### Prompt Design
Prompts are designed to strictly govern the behavior of the agent:
1. **Extraction Prompt**: Focuses on analyzing the conversation to emit one of four intents (`clarify`, `off_topic`, `search`, `compare`) and extract `search_keywords`.
2. **Generation Prompt**: Grounded heavily with rules such as "NEVER recommend anything outside the SHL catalog." We leverage Gemini's structured output to ensure the response strictly matches the required JSON format (`reply`, `recommendations`, `end_of_conversation`).

### Evaluation Approach
We evaluated the agent using the provided 10 public conversation traces. We run the FastAPI endpoint locally and pass the multi-turn arrays incrementally.
- **What didn't work**: Initially, providing the entire 377-item catalog directly into the context without retrieval caused occasional hallucination of features and slower response times.
- **Improvements**: Moving to the TF-IDF retrieval step improved precision and constrained the model to the most relevant items, significantly reducing hallucination and guaranteeing compliance with the 8-turn cap and 30s timeout.
