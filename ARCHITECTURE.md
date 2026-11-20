# System Architecture & Technical Design Specification
## Multimodal Prescriptive Maintenance (PdM) System with HITL RAG

---

## 1. Executive Summary & Problem Domain

Traditional industrial Predictive Maintenance (PdM) systems answer two primary questions:
1. **"When will the asset fail?"** (Remaining Useful Life / RUL regression)
2. **"What component is failing?"** (Fault classification)

While mathematically sound, these outputs leave a critical operational void. Maintenance technicians and plant operators are confronted with raw sensor anomaly spikes and technical error codes, forcing them to manually parse hundreds of pages of unindexed equipment manuals and Standard Operating Procedures (SOPs) under high-stress downtime conditions.

This platform bridges that operational gap by transforming predictive signals into **prescriptive, automated remediation workflows**. The system couples deterministic, high-recall discriminative machine learning models with a private, air-gapped Retrieval-Augmented Generation (RAG) assistant, Explainable AI (SHAP) telemetry attribution, and a closed-loop Human-in-the-Loop (HITL) feedback system.

---

## 2. High-Level Architectural Topology

```
+---------------------------------------------------------------------------------------------------+
|                                      DATA INGESTION & CLIENT                                      |
|                                                                                                   |
|  +---------------------------+    HTTP/REST     +-----------------------------------------------+  |
|  |  Industrial Telemetry     | ---------------> |              Flask API Gateway                |  |
|  |  (Sensors, SCADA, Logs)   |                  |  - Schema Validation (Boundary Checks)        |  |
|  +---------------------------+                  |  - Route Orchestration & Error Boundaries     |  |
+-------------------------------------------------+-----------------------+-------------------------+
                                                                          |
                                      +-----------------------------------+-----------------------------------+
                                      |                                                                       |
                                      v                                                                       v
+-------------------------------------------------------------------+   +-------------------------------------------------------------------+
|               SUBSYSTEM A: PREDICTIVE INFERENCE ENGINE            |   |               SUBSYSTEM B: PRESCRIPTIVE RAG ENGINE                |
|                                                                   |   |                                                                   |
|  +-------------------------------------------------------------+  |   |  +-------------------------------------------------------------+  |
|  | Feature Engineering & Preprocessing Pipeline                |  |   |  | Technical Documentation & SOP Ingestion Pipeline            |  |
|  | (Rolling Aggregations, MinMax Scaling, Trend Differencing)  |  |   |  | (Markdown/PDF Manuals, Recursive Semantic Chunking)         |  |
|  +------------------------------+------------------------------+  |   |  +------------------------------+------------------------------+  |
|                                 |                                 |   |                                 |                                 |
|                 +---------------+---------------+                 |   |                                 v                                 |
|                 |                               |                 |   |  +-------------------------------------------------------------+  |
|                 v                               v                 |   |  | Vector Database: ChromaDB (Persistent Disk Storage)         |  |
|  +----------------------------+   +----------------------------+  |   |  | Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-d)  |  |
|  | RUL Regression Model       |   | Multi-Class Classifier     |  |   |  +------------------------------+------------------------------+  |
|  | LightGBM / XGBoost Regr.   |   | Cost-Weighted Gradient     |  |   |                                 |                                 |
|  | RMSE: 15.82 | R²: 0.85     |   | Boosted Trees (95% Recall) |  |   |                                 | Cosine Similarity               |
|  +--------------+-------------+   +--------------+-------------+  |   |                                 | Retrieval (Top-K)               |
|                 |                                |                |   |                                 v                                 |
|                 +---------------+----------------+                |   |  +-------------------------------------------------------------+  |
|                                 |                                 |   |  | RAG Context Synthesis Engine (LangChain Orchestrator)       |  |
|                                 v                                 |   |  | Prompt Template: Fault Metadata + SHAP Attributions + SOPs  |  |
|  +-------------------------------------------------------------+  |   |  +------------------------------+------------------------------+  |
|  | Explainable AI (XAI) Attribution                           |  |   |                                 |                                 |
|  | TreeSHAP: Generates Local Feature Importance Vectors       |  |   |                                 v                                 |
|  +------------------------------+------------------------------+  |   |  +-------------------------------------------------------------+  |
|                                 |                                 |   |  | Local-First LLM Inference (Ollama / Quantized Local Models)  |  |
|                                 +---------------------------------+-> |  | Structured Prescriptive Remediation & Part Checklists        |  |
|                                                                       +---------------------------------+---------------------------------+
|                                                                                                         |
+---------------------------------------------------------------------------------------------------------+
                                                                          |
                                                                          v
+---------------------------------------------------------------------------------------------------+
|                                 SUBSYSTEM C: HUMAN-IN-THE-LOOP (HITL)                             |
|                                                                                                   |
|  +---------------------------------------+         +-------------------------------------------+  |
|  | Operator Verification Interface       | <-----> | Feedback Ingestion Endpoint               |  |
|  | (Accept, Override, Flag False Alerts) |         | POST /api/feedback                        |  |
|  +---------------------------------------+         +---------------------+---------------------+  |
|                                                                          |                        |
|                                                                          v                        |
|                                                    +-------------------------------------------+  |
|                                                    | Feedback Persistence & Audit Trail        |  |
|                                                    | (Continuous Learning & Calibration Store) |  |
|                                                    +-------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Core Architectural Subsystems

### 3.1. Telemetry Ingestion & Feature Engineering
Industrial sensor time-series data requires rigorous deterministic cleaning before hitting non-linear estimators:
- **Feature Pipeline**: Implements temporal sliding windows to derive moving averages, standard deviations, and signal derivatives for dynamic sensor feeds (vibration, bearing temperature, fluid pressure).
- **Imbalance Mitigation**: Industrial failures represent $<1\%$ of raw operational logs. The training lifecycle combines Synthetic Minority Over-sampling Technique (SMOTE) with cost-sensitive loss functions ($C_{\text{FN}} \gg C_{\text{FP}}$) to prioritize false-negative minimization.

### 3.2. Deterministic Dual-Model Predictive Engine
To prevent generative hallucinations from causing catastrophic physical errors, predictive decisions remain strictly non-probabilistic:
- **Remaining Useful Life (RUL) Regressor**: Built with LightGBM and XGBoost, achieving an **RMSE of 15.82** and **$R^2$ of 0.85**. Outputs a scalar float representing expected operational hours remaining before reaching critical failure boundaries.
- **Fault Mode Classifier**: Multi-class gradient-boosted classifier identifying failure sub-assemblies (e.g., *Bearing Inner Race Defect*, *Stator Insulation Breakdown*, *Cavitation*). Achieves **up to 95% Recall** on critical failure modes.

### 3.3. Explainability Layer (TreeSHAP)
Black-box outputs are rejected by plant engineers. The inference pipeline computes TreeSHAP (SHapley Additive exPlanations) values on the fly:
- Computes the base marginal contribution of each telemetry feature against the current prediction.
- Identifies and ranks the **Top-3 Anomaly Drivers** (e.g., `Sensor_11 > 2.4 sigma above nominal operating baseline`).
- Pipes these feature attributions directly into the context payload of Subsystem B to ground the prescriptive copilot.

### 3.4. Prescriptive Knowledge Retrieval (Air-Gapped RAG)
Once the predictive engine flags an imminent failure mode, the prescriptive copilot activates:
- **Vector Database**: ChromaDB embedded on disk, indexing technical manuals, OEM service bulletins, and maintenance work orders.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense semantic vectors. Fast, lightweight, and optimized for CPU inference.
- **Prompt Engineering & Context Grounding**:
```text
[SYSTEM CONTEXT]
You are an industrial prescriptive maintenance assistant.
Base your instructions strictly on the provided OEM technical manuals.

[DIAGNOSTIC TELEMETRY]
- Target Asset: Hydraulic Pump Unit #04
- Remaining Useful Life: 14.2 Hours (CRITICAL)
- Predicted Failure Mode: Bearing Raceway Degradation (Class 3)
- Key Anomalous Indicators: Sensor_07 (Vibration RMS: +3.2 std), Sensor_11 (Thermal: +2.8 std)

[RETRIEVED OEM MANUAL EXCERPTS]
{Retrieved context from ChromaDB}

[OUTPUT INSTRUCTION]
Provide:
1. Root cause summary.
2. Immediate lockout/tagout (LOTO) safety protocol.
3. Step-by-step remediation guide.
4. Required OEM replacement part numbers and tooling.
```
- **Local LLM Orchestration**: Driven by LangChain interfaces interacting with a local **Ollama** runtime (running quantized models like `Llama-3` or `Phi-3`). Zero external network egress guarantees enterprise data privacy.

### 3.5. Human-in-the-Loop (HITL) Closed-Loop Adaptation
No industrial AI system should operate fully autonomous without human oversight:
- The UI exposes structured action buttons: **Approve SOP**, **Override Diagnosis**, and **Flag Incorrect Alert**.
- Technician inputs are routed to `/api/feedback`, recording:
  - Input telemetry hash.
  - Model diagnostic output vs. Technician's ground truth override.
  - Technician notes and actual corrective actions taken.
- This creates an auditable historical dataset for drift detection and scheduled model re-training.

---

## 4. API Interface & Data Contracts

### `POST /api/predict`
Evaluates raw sensor arrays against discriminative models and returns predicted degradation and failure modes.

#### Request Payload:
```json
{
  "asset_id": "PUMP-WEST-042",
  "timestamp": "2026-11-20T14:32:00Z",
  "telemetry": {
    "sensor_1": 412.35,
    "sensor_2": 1.42,
    "sensor_3": 58.11,
    "sensor_4": 1204.0
  }
}
```

#### Response Payload:
```json
{
  "status": "success",
  "asset_id": "PUMP-WEST-042",
  "prediction": {
    "remaining_useful_life_hours": 15.82,
    "urgency": "CRITICAL",
    "fault_detected": true,
    "fault_class": "Bearing Raceway Degradation",
    "confidence": 0.942
  },
  "explainability": {
    "top_contributors": [
      {"feature": "sensor_2", "shap_value": 0.412, "description": "Vibration Spike"},
      {"feature": "sensor_1", "shap_value": 0.285, "description": "Thermal Runaway"}
    ]
  }
}
```

---

### `POST /api/prescribe`
Combines predictive output with vector-retrieved OEM procedures to generate an actionable repair plan.

#### Request Payload:
```json
{
  "asset_id": "PUMP-WEST-042",
  "fault_class": "Bearing Raceway Degradation",
  "top_contributors": ["sensor_2", "sensor_1"]
}
```

#### Response Payload:
```json
{
  "status": "success",
  "prescription": {
    "summary": "Immediate bearing lubrication failure detected leading to raceway micro-pitting.",
    "loto_procedure": "LOTO-ISO-09: Isolate 480V main bus and depressurize hydraulic line 4B.",
    "remediation_steps": [
      "De-energize pump unit and attach safety lockouts.",
      "Remove bearing housing casing using 14mm hex socket.",
      "Inspect raceway for pitting; flush debris with solvent-44.",
      "Replace bearing unit with OEM Part #SKF-6205-2RS."
    ],
    "required_parts": ["SKF-6205-2RS", "Nitril O-Ring 45mm x 3mm"],
    "retrieved_sources": [
      {"manual": "OEM-Pump-Maintenance-Rev4.pdf", "section": "Section 8.2: Bearing Replacement"}
    ]
  }
}
```

---

### `POST /api/feedback`
Captures human technician domain verification.

#### Request Payload:
```json
{
  "asset_id": "PUMP-WEST-042",
  "prediction_id": "pred_88294a_772",
  "technician_id": "TECH_J_DOE",
  "feedback_action": "OVERRIDE",
  "corrected_fault_class": "Cavitation Bubbling",
  "technician_notes": "Vibration was caused by vapor cavity implosion due to upstream valve blockage, not bearing wear."
}
```

#### Response Payload:
```json
{
  "status": "recorded",
  "feedback_id": "fb_00921_ack",
  "audit_logged": true
}
```

---

## 5. Production Invariants & Reliability Guarantees

1. **Air-Gapped & Zero Data Egress**: Sensitive operational diagrams and facility logs never leave the internal perimeter. Embedding generation, vector similarity searches, and LLM inference operate locally on premise.
2. **Deterministic Fallbacks**: If the Ollama runtime is unreachable or experiences memory starvation, the system gracefully falls back to deterministic rule-based lookups from a pre-indexed static JSON SOP repository. Predictive scoring is completely decoupled from generative synthesis.
3. **Reproducibility**: All model pipelines, embedding indices, and inference environments are containerized via Docker with fixed software dependency trees, ensuring identical execution across bare-metal plant servers or cloud instances.
4. **Resilient Numerical Boundaries**: Inputs are strictly checked against mathematical boundaries. Missing telemetry values are imputed using temporal forward-fill estimators rather than zero-padding, preventing artificial shock-spikes into the model.

---

## 6. Known Limitations & Architectural Trade-offs

A transparent evaluation of architectural decisions and their future enterprise evolution:

| Architectural Component | Current Implementation | Trade-off / Technical Rationale | Production Evolution Roadmap |
| :--- | :--- | :--- | :--- |
| **API Web Server** | Flask with Gunicorn (Sync Workers) | Chosen for low overhead, predictable synchronous memory profiling with ML models, and rapid validation. | Migrate to **FastAPI / Litestar (ASGI)** to support native asynchronous WebSockets for real-time high-throughput SCADA telemetry streams. |
| **Vector Storage** | ChromaDB (Embedded SQLite) | Zero configuration; runs inside the container without external network dependencies. | Transition to a distributed instance of **Qdrant** or **pgvector (PostgreSQL)** for multi-tenant plant isolation and distributed shard scaling. |
| **Task Queue & Concurrency** | Synchronous Request-Response | Keeps the deployment self-contained in a single container without external broker dependencies. | Introduce **Redis Streams / Celery** to decouple heavy generative RAG synthesis from the sub-millisecond telemetry ingestion path. |
| **Model Inference Serving** | Direct in-memory Python (`joblib`) | Eliminates network hops and serialization overhead for small-to-medium sensor arrays. | Export models to **ONNX Runtime** or serve via **Triton Inference Server** for dynamic batching and GPU acceleration. |
| **Data Ingestion Invariants** | Discrete HTTP REST Batches | Well-suited for discrete equipment evaluation cycles and scheduled edge reporting. | Implement an **Apache Kafka** or **MQTT** broker layer to handle raw, continuous millisecond-frequency sensor streaming. |

---

## 7. Local Development & Testing Workflow

### Automated Test Execution
Run the automated test suite covering feature engineering pipelines, model scoring consistency, and API boundary contracts:
```bash
# Execute unit and integration tests
pytest -v --tb=short
```

### Docker Container Deployment
```bash
# Build production Docker image
docker build -t pdm-system:latest .

# Run container with volume-mapped persistent stores
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  --name pdm-production \
  pdm-system:latest

# Verify service health status
curl -i http://localhost:5000/health
```
