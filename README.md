# Hybrid Semantic Router

![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)
![Embeddings](https://img.shields.io/badge/Vector-Embeddings-blueviolet.svg)

A high-performance query routing layer that sits in front of LLM applications. Instead of passing every user query to an expensive LLM to determine intent, this system uses ultra-fast local embeddings and keyword matching to route queries to the correct downstream agent, tool, or prompt template.

## 🎯 The Problem
Routing user intent using an LLM (e.g., zero-shot classification via OpenAI) adds 500ms-1s of latency and costs money per query. 

## ⚡ The Solution
The **Hybrid Semantic Router** achieves sub-50ms routing by combining semantic similarity (Dense Vectors) with keyword matching (Sparse Vectors / BM25).

## 🚀 Features

- **Hybrid Search:** Fuses cosine similarity from a local embedding model (e.g., `all-MiniLM-L6-v2` via ONNX) with BM25 keyword scoring for high-accuracy intent classification.
- **Route Definitions:** Define routes using a few utterance examples. The router calculates the centroid of these examples to define the route's semantic space.
- **Dynamic Fallback:** If the similarity score is below a predefined threshold, the query is routed to a general fallback LLM.
- **Extremely Fast:** Built with FastAPI and optimized ONNX runtime. Capable of handling thousands of requests per second on CPU.

## 🛠️ Usage Example

```python
from semantic_router import Route, HybridRouter

# 1. Define Routes
sales_route = Route(
    name="sales_inquiry",
    utterances=["How much does the enterprise plan cost?", "I want to talk to sales", "Pricing details"]
)

support_route = Route(
    name="tech_support",
    utterances=["My server is down", "How do I reset my password?", "Getting a 500 error"]
)

# 2. Initialize Router
router = HybridRouter(routes=[sales_route, support_route])

# 3. Route a Query
decision = router("I forgot my login password.")
print(decision.name) # Output: tech_support
```
