# TruLens Supported Strategies in the Unified Evaluation Workflow

This document maps TruLens capabilities to the Unified Evaluation Workflow, using a three-tier classification system to distinguish native functionality from third-party integrations.

## Classification Framework

### ✅ Natively Supported
Steps that meet ALL of the following requirements:
- Available immediately after installing TruLens (`pip install trulens`)
- Requires only import statements and minimal configuration (≤2 lines)
- No external dependencies beyond TruLens packages
- No custom implementation or glue code required

### 🔌 Supported via Third-Party Integration
Steps that meet ALL of the following requirements:
- Requires installing ≥1 external package(s) (e.g., `openai`, `langchain`, `transformers`)
- Requires glue code (typically ≤10 lines)
- Has documented integration pattern or official example
- Functionality enabled through third-party tools rather than TruLens alone

### ❌ Not Supported
Features that are not available in TruLens, either natively or through documented integrations.

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: PyPI Packages** | ✅ Natively Supported | TruLens is installable via `pip install trulens` from PyPI. All core functionality is available through PyPI packages. |
| **Strategy 2: Git Clone** | ✅ Natively Supported | TruLens can be cloned from GitHub and installed from source using `pip install -e .` for development purposes. |
| **Strategy 3: Container Images** | ❌ Not Supported | TruLens does not provide prebuilt Docker or OCI container images. Users must create their own containers if needed. |
| **Strategy 4: Binary Packages** | ❌ Not Supported | TruLens does not distribute standalone executable binaries. |
| **Strategy 5: Node Package** | ❌ Not Supported | TruLens is a Python-based framework and does not provide Node.js/npm packages. |

### Step B: Service Authentication

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Evaluation Platform Authentication** | ❌ Not Supported | TruLens does not require registration or authentication with an evaluation platform service. It operates locally or with user-provided databases. |
| **Strategy 2: API Provider Authentication** | 🔌 Supported via Third-Party Integration | TruLens feedback providers (OpenAI, Google, Bedrock, LiteLLM, etc.) support authentication via environment variables and API keys. Requires installing provider-specific packages (e.g., `pip install trulens-providers-openai`) and configuring credentials. |
| **Strategy 3: Repository Authentication** | 🔌 Supported via Third-Party Integration | TruLens integrates with Hugging Face models and datasets. Requires `transformers` or `datasets` packages and token-based authentication through environment variables or CLI login. |

---

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Model-as-a-Service (Remote Inference)** | 🔌 Supported via Third-Party Integration | TruLens feedback providers support remote inference through API endpoints (OpenAI, Google, Azure OpenAI, Bedrock, LiteLLM). Requires provider-specific packages (e.g., `openai`, `google-genai`) and API key configuration. Applications using these services can be instrumented with TruLens wrappers. |
| **Strategy 2: Model-in-Process (Local Inference)** | 🔌 Supported via Third-Party Integration | TruLens can instrument applications that load and run models locally. The `HuggingfaceLocal` provider enables local model execution for feedback functions. Requires `transformers`, `torch`, or similar ML libraries plus model loading code. |
| **Strategy 3: Algorithm Implementation (In-Memory Structures)** | 🔌 Supported via Third-Party Integration | TruLens can instrument custom retrieval algorithms (e.g., FAISS, ChromaDB, BM25) when they are part of an instrumented RAG application. Requires the user to implement or install the algorithm, then instrument with TruLens decorators. TruLens does not provide algorithm implementations itself. |
| **Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** | 🔌 Supported via Third-Party Integration | TruLens supports instrumenting agents through framework integrations. `TruGraph` wraps LangGraph agents (requires `langgraph` package). Interactive loop execution tracking with state transitions available through framework-specific wrappers. |

### Step B: Benchmark Preparation (Inputs)

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Benchmark Dataset Preparation (Offline)** | 🔌 Supported via Third-Party Integration | TruLens benchmark module includes dataset loading capabilities (e.g., `beir_loader.py`). Supports evaluating applications on pre-existing datasets. Requires dataset libraries (e.g., `datasets`, `beir`) and loading/transformation code. |
| **Strategy 2: Synthetic Data Generation (Generative)** | 🔌 Supported via Third-Party Integration | TruLens provides `GenerateTestSet` class that generates synthetic test cases. Requires an LLM provider (OpenAI, etc.) to generate test prompts. Configuration includes breadth/depth parameters and optional few-shot examples. |
| **Strategy 3: Simulation Environment Setup (Simulated)** | ❌ Not Supported | TruLens does not provide 3D simulation environments or physics-based environment setup. It is focused on LLM/RAG evaluation, not robotics or game simulations. |
| **Strategy 4: Production Traffic Sampling (Online)** | ✅ Natively Supported | TruLens supports logging production traffic in real-time by instrumenting applications with `TruBasicApp`, `TruChain`, or other wrappers. Captures all inputs/outputs during execution with minimal code (2-3 lines to wrap app). |

### Step C: Benchmark Preparation (References)

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Judge Preparation** | 🔌 Supported via Third-Party Integration | TruLens provides pre-configured LLM-as-judge feedback functions through multiple providers (OpenAI, Google, Bedrock, LiteLLM, Langchain). Requires provider packages and API credentials. Users can also create custom feedback functions by implementing the provider interface. |
| **Strategy 2: Ground Truth Preparation** | ✅ Natively Supported | TruLens includes `GroundTruthAgreement` provider for evaluating against ground truth data. The benchmark framework supports ground truth labels and reference materials. Minimal configuration required after installing TruLens. |

---

## Phase II: Execution (The Run)

### Step A: SUT Invocation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Batch Inference** | ✅ Natively Supported | TruLens supports batch evaluation by instrumenting applications and running them on multiple inputs. The `TruBenchmarkExperiment` class enables running feedback functions across datasets with parallel execution support. Minimal setup after installing TruLens. |
| **Strategy 2: Interactive Loop** | 🔌 Supported via Third-Party Integration | TruLens can instrument and trace interactive agent loops, particularly through LangGraph integration (`TruGraph`). Requires `langgraph` package. It captures state transitions, tool calls, and multi-step reasoning. Inline evaluations provide real-time feedback during execution. |
| **Strategy 3: Arena Battle** | ❌ Not Supported | TruLens does not provide native pairwise comparison or arena battle functionality where the same input is sent to multiple SUTs simultaneously for direct comparison. Users must implement this manually. |
| **Strategy 4: Production Streaming** | ✅ Natively Supported | TruLens can instrument production applications and capture real-time metrics as traffic flows through the system. It logs inputs, outputs, and feedback scores for live production traffic. Simple wrapper integration (2-3 lines). |

---

## Phase III: Assessment (The Score)

### Step A: Individual Scoring

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Deterministic Measurement** | ✅ Natively Supported | TruLens feedback implementations include deterministic metrics like exact matching, string equality checks, and token-based similarity measurements. Custom deterministic feedback functions can be easily implemented with minimal code. |
| **Strategy 2: Embedding Measurement** | 🔌 Supported via Third-Party Integration | TruLens provides `Embeddings` feedback provider for semantic similarity calculations. Requires embedding model packages (e.g., `sentence-transformers`) and model configuration. Supports BERTScore and custom embedding-based comparisons. |
| **Strategy 3: Subjective Measurement** | 🔌 Supported via Third-Party Integration | Core strength of TruLens. Provides extensive LLM-as-judge capabilities through multiple providers (OpenAI, Google, Bedrock, LiteLLM, Langchain). Requires provider packages and API keys. Includes pre-built feedback functions for relevance, coherence, groundedness, and custom subjective evaluations. |
| **Strategy 4: Performance Measurement** | ✅ Natively Supported | TruLens automatically tracks usage metrics including token counts (prompt/completion), request counts, latency (through span timing), and cost calculations in USD. These metrics are captured from LLM spans during execution with no additional configuration. |

### Step B: Collective Aggregation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Score Aggregation** | ✅ Natively Supported | TruLens supports aggregating individual scores through feedback aggregation functions (mean, weighted average, custom aggregators). The dashboard displays aggregate metrics at the application and benchmark level. Built into core functionality. |
| **Strategy 2: Uncertainty Quantification** | ❌ Not Supported | TruLens does not provide native bootstrap resampling or Prediction-Powered Inference (PPI) for confidence interval estimation. Statistical uncertainty quantification must be implemented by users. |

---

## Phase IV: Reporting (The Output)

### Step A: Insight Presentation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Execution Tracing** | ✅ Natively Supported | TruLens provides comprehensive execution tracing through OpenTelemetry-based instrumentation. The dashboard shows detailed step-by-step traces with tool calls, intermediate states, span attributes, and decision paths. Built into core functionality with `run_dashboard()`. |
| **Strategy 2: Subgroup Analysis** | ✅ Natively Supported | TruLens supports metadata logging on records, enabling stratification and filtering of evaluation results by custom dimensions (e.g., `record_metadata=dict(prompt_category=category)`). Dashboard allows filtering by metadata. Simple 1-line metadata configuration. |
| **Strategy 3: Chart Generation** | ✅ Natively Supported | TruLens dashboard includes visual representations of evaluation results, including feedback score distributions, performance trends, and comparative visualizations across application versions. Available immediately via `run_dashboard()`. |
| **Strategy 4: Dashboard Creation** | ✅ Natively Supported | TruLens ships with a built-in Streamlit dashboard (`run_dashboard()`) that provides interactive web interface with leaderboard view, evaluation results, trace explorer, and filterable tables. Also provides embeddable Streamlit components. Single-line launch. |
| **Strategy 5: Leaderboard Publication** | ✅ Natively Supported | TruLens provides a local leaderboard view in its dashboard comparing different application versions and configurations. Accessible via built-in dashboard. Note: Does not support submitting to external public leaderboards. |
| **Strategy 6: Regression Alerting** | ❌ Not Supported | TruLens does not provide native regression detection or automated alerting when metrics fall below thresholds. Users must implement monitoring and alerting logic externally. |

---

## Summary

### Strategy Support Breakdown

**✅ Natively Supported (14 strategies):**
- Installation via PyPI and Git
- Production traffic sampling and streaming
- Ground truth preparation
- Batch inference
- Deterministic measurement and performance tracking
- Score aggregation
- Complete dashboard, tracing, charts, subgroup analysis, and leaderboard

**🔌 Supported via Third-Party Integration (12 strategies):**
- API provider and repository authentication
- Model-as-a-Service and Model-in-Process inference
- Algorithm implementation instrumentation
- Agent/policy instantiation
- Benchmark dataset loading
- Synthetic data generation
- Judge preparation (LLM-as-judge)
- Interactive loop execution
- Embedding and subjective measurement

**❌ Not Supported (8 strategies):**
- Container images, binary packages, Node packages
- Evaluation platform authentication
- 3D simulation environments
- Arena battle/pairwise comparison
- Statistical uncertainty quantification
- Automated regression alerting

### Key Insights

**Native Strengths:**
TruLens excels at observability and evaluation infrastructure with minimal setup. Core capabilities (tracing, dashboards, aggregation, production monitoring) require only TruLens installation and 1-2 lines of code.

**Integration Model:**
Most evaluation capabilities (LLM-as-judge, embeddings, framework integrations) follow a consistent pattern: install provider package + configure credentials + minimal glue code. TruLens provides the instrumentation and evaluation framework; users bring their model/data providers.

**Architectural Philosophy:**
TruLens is designed as an instrumentation and evaluation layer that wraps existing ML infrastructure rather than reimplementing it. This explains why most SUT and scoring strategies require third-party integrations.

---

**Last Updated**: 2025-12-16  
**TruLens Version Reviewed**: Current main branch
