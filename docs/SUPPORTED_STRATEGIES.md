# TruLens Supported Strategies in the Unified Evaluation Workflow

This document maps TruLens capabilities to the [Unified Evaluation Workflow](https://github.com/truera/trulens), identifying which strategies are natively supported in a full TruLens installation.

A strategy is marked as **SUPPORTED** only if TruLens provides it out-of-the-box after full installation—meaning it can be executed directly without implementing custom modules or integrating external libraries beyond what's included in the TruLens ecosystem.

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: PyPI Packages** | ✅ SUPPORTED | TruLens is installable via `pip install trulens` from PyPI. All core functionality is available through PyPI packages. |
| **Strategy 2: Git Clone** | ✅ SUPPORTED | TruLens can be cloned from GitHub and installed from source using `pip install -e .` for development purposes. |
| **Strategy 3: Container Images** | ❌ UNSUPPORTED | TruLens does not provide prebuilt Docker or OCI container images. Users must create their own containers if needed. |
| **Strategy 4: Binary Packages** | ❌ UNSUPPORTED | TruLens does not distribute standalone executable binaries. |
| **Strategy 5: Node Package** | ❌ UNSUPPORTED | TruLens is a Python-based framework and does not provide Node.js/npm packages. |

### Step B: Service Authentication

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Evaluation Platform Authentication** | ❌ UNSUPPORTED | TruLens does not require registration or authentication with an evaluation platform service. It operates locally or with user-provided databases. |
| **Strategy 2: API Provider Authentication** | ✅ SUPPORTED | TruLens feedback providers (OpenAI, Google, Bedrock, LiteLLM, etc.) support authentication via environment variables and API keys for remote model inference. |
| **Strategy 3: Repository Authentication** | ✅ SUPPORTED | TruLens integrates with Hugging Face models and datasets, supporting token-based authentication through environment variables or CLI login for accessing models and datasets. |

---

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Model-as-a-Service (Remote Inference)** | ✅ SUPPORTED | TruLens feedback providers support remote inference through API endpoints (OpenAI, Google, Azure OpenAI, Bedrock, LiteLLM). Applications using these services can be instrumented and evaluated. |
| **Strategy 2: Model-in-Process (Local Inference)** | ✅ SUPPORTED | TruLens can instrument applications that load and run models locally (LLMs via Hugging Face transformers, embedding models, etc.). The `HuggingfaceLocal` provider enables local model execution for feedback functions. |
| **Strategy 3: Algorithm Implementation (In-Memory Structures)** | ⚠️ PARTIAL | TruLens can instrument custom retrieval algorithms (e.g., vector search, BM25) when they are part of an instrumented RAG application, but does not provide native implementations of these algorithms itself. |
| **Strategy 4: Policy/Agent Instantiation (Stateful Controllers)** | ✅ SUPPORTED | TruLens supports instrumenting agents through LangGraph integration (`TruGraph`) and provides interactive loop execution tracking with state transitions. |

### Step B: Benchmark Preparation (Inputs)

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Benchmark Dataset Preparation (Offline)** | ✅ SUPPORTED | TruLens benchmark module includes dataset loading capabilities (e.g., `beir_loader.py`) and supports evaluating applications on pre-existing datasets. Users can load, transform, and format benchmark datasets for evaluation. |
| **Strategy 2: Synthetic Data Generation (Generative)** | ✅ SUPPORTED | TruLens provides `GenerateTestSet` class that generates synthetic test cases with configurable breadth and depth, creating test prompts across multiple categories. |
| **Strategy 3: Simulation Environment Setup (Simulated)** | ❌ UNSUPPORTED | TruLens does not provide 3D simulation environments or physics-based environment setup. It is focused on LLM/RAG evaluation, not robotics or game simulations. |
| **Strategy 4: Production Traffic Sampling (Online)** | ✅ SUPPORTED | TruLens supports logging production traffic in real-time by instrumenting applications and capturing all inputs/outputs during execution. Data can be sampled from live inference traffic. |

### Step C: Benchmark Preparation (References)

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Judge Preparation** | ✅ SUPPORTED | TruLens provides pre-configured LLM-as-judge feedback functions through multiple providers (OpenAI, Google, Bedrock, LiteLLM, Langchain). Users can also create custom feedback functions and load pre-trained judge models. |
| **Strategy 2: Ground Truth Preparation** | ✅ SUPPORTED | TruLens includes `GroundTruthAgreement` provider for evaluating against ground truth data. The benchmark framework supports ground truth labels and reference materials for evaluation. |

---

## Phase II: Execution (The Run)

### Step A: SUT Invocation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Batch Inference** | ✅ SUPPORTED | TruLens supports batch evaluation by instrumenting applications and running them on multiple inputs. The `TruBenchmarkExperiment` class enables running feedback functions across datasets with parallel execution support. |
| **Strategy 2: Interactive Loop** | ✅ SUPPORTED | TruLens can instrument and trace interactive agent loops, particularly through LangGraph integration (`TruGraph`). It captures state transitions, tool calls, and multi-step reasoning. Inline evaluations provide real-time feedback during execution. |
| **Strategy 3: Arena Battle** | ❌ UNSUPPORTED | TruLens does not provide native pairwise comparison or arena battle functionality where the same input is sent to multiple SUTs simultaneously for direct comparison. Users must implement this manually. |
| **Strategy 4: Production Streaming** | ✅ SUPPORTED | TruLens can instrument production applications and capture real-time metrics as traffic flows through the system. It logs inputs, outputs, and feedback scores for live production traffic. |

---

## Phase III: Assessment (The Score)

### Step A: Individual Scoring

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Deterministic Measurement** | ✅ SUPPORTED | TruLens feedback implementations include deterministic metrics like exact matching, string equality checks, and token-based similarity measurements. Custom deterministic feedback functions can be easily implemented. |
| **Strategy 2: Embedding Measurement** | ✅ SUPPORTED | TruLens provides `Embeddings` feedback provider for semantic similarity calculations using embedding models. Supports BERTScore and custom embedding-based comparisons through the embeddings module. |
| **Strategy 3: Subjective Measurement** | ✅ SUPPORTED | Core strength of TruLens. Provides extensive LLM-as-judge capabilities through multiple providers (OpenAI, Google, Bedrock, LiteLLM, Langchain). Includes pre-built feedback functions for relevance, coherence, groundedness, and custom subjective evaluations. |
| **Strategy 4: Performance Measurement** | ✅ SUPPORTED | TruLens automatically tracks usage metrics including token counts (prompt/completion), request counts, latency (through span timing), and cost calculations in USD. These metrics are captured from LLM spans during execution. |

### Step B: Collective Aggregation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Score Aggregation** | ✅ SUPPORTED | TruLens supports aggregating individual scores through feedback aggregation functions (mean, weighted average, custom aggregators). The dashboard displays aggregate metrics at the application and benchmark level. |
| **Strategy 2: Uncertainty Quantification** | ❌ UNSUPPORTED | TruLens does not provide native bootstrap resampling or Prediction-Powered Inference (PPI) for confidence interval estimation. Statistical uncertainty quantification must be implemented by users. |

---

## Phase IV: Reporting (The Output)

### Step A: Insight Presentation

| Strategy | Status | Justification |
|----------|--------|---------------|
| **Strategy 1: Execution Tracing** | ✅ SUPPORTED | TruLens provides comprehensive execution tracing through OpenTelemetry-based instrumentation. The dashboard shows detailed step-by-step traces with tool calls, intermediate states, span attributes, and decision paths. |
| **Strategy 2: Subgroup Analysis** | ✅ SUPPORTED | TruLens supports metadata logging on records, enabling stratification and filtering of evaluation results by custom dimensions (e.g., `record_metadata=dict(prompt_category=category)`). Dashboard allows filtering by metadata. |
| **Strategy 3: Chart Generation** | ✅ SUPPORTED | TruLens dashboard includes visual representations of evaluation results, including feedback score distributions, performance trends, and comparative visualizations across application versions. |
| **Strategy 4: Dashboard Creation** | ✅ SUPPORTED | TruLens ships with a built-in Streamlit dashboard (`run_dashboard()`) that provides interactive web interface with leaderboard view, evaluation results, trace explorer, and filterable tables. Also provides embeddable Streamlit components. |
| **Strategy 5: Leaderboard Publication** | ⚠️ PARTIAL | TruLens provides a local leaderboard view in its dashboard comparing different application versions and configurations. However, it does not support submitting results to external public leaderboards. |
| **Strategy 6: Regression Alerting** | ❌ UNSUPPORTED | TruLens does not provide native regression detection or automated alerting when metrics fall below thresholds. Users must implement monitoring and alerting logic externally. |

---

## Summary

### Fully Supported Phases
- **Phase I (Specification)**: Strong support for SUT preparation, benchmark datasets, synthetic generation, and judge/ground truth preparation
- **Phase III (Assessment)**: Comprehensive support for all individual scoring types and basic aggregation
- **Phase IV (Reporting)**: Excellent support for tracing, visualization, and dashboard presentation

### Partially Supported
- **Phase 0 (Provisioning)**: Supports Python-based installation and API authentication but lacks containerization
- **Phase II (Execution)**: Supports batch and interactive execution but lacks arena battle mode

### Key Gaps
- Container images and binary distributions
- Evaluation platform-specific authentication
- Simulation environments for robotics/gaming
- Pairwise arena comparisons
- Statistical uncertainty quantification
- Automated regression alerting

---

**Last Updated**: 2025-12-12  
**TruLens Version Reviewed**: Current main branch
