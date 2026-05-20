# Flower — Technical Documentation

> **Version**: 1.30.0  
> **License**: Apache 2.0  
> **Maintainer**: Flower Labs GmbH  
> **Website**: https://flower.ai  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Architecture](#3-architecture)
4. [Core Components](#4-core-components)
   - 4.1 [Python Framework (`flwr`)](#41-python-framework-flwr)
   - 4.2 [Federated Datasets (`flwr-datasets`)](#42-federated-datasets-flwr-datasets)
   - 4.3 [Flower Intelligence (Multi-platform SDKs)](#43-flower-intelligence-multi-platform-sdks)
   - 4.4 [Flower Baselines](#44-flower-baselines)
5. [Data Models & Protocols](#5-data-models--protocols)
6. [Federated Learning Strategies](#6-federated-learning-strategies)
7. [CLI Reference](#7-cli-reference)
8. [Configuration](#8-configuration)
9. [Database Schema](#9-database-schema)
10. [Transport Layer](#10-transport-layer)
11. [Security](#11-security)
12. [Testing](#12-testing)
13. [Build & Deployment](#13-build--deployment)
14. [Development Guide](#14-development-guide)
15. [Examples & Baselines](#15-examples--baselines)

---

## 1. Project Overview

Flower (`flwr`) is an open-source, production-grade **federated AI framework** designed for building systems where machine learning training happens across distributed devices or data silos — without centralizing raw data. It originated from research at the University of Oxford and is now maintained by Flower Labs GmbH.

### Design Principles

| Principle | Description |
|-----------|-------------|
| **Customizable** | Supports a wide range of FL configurations for different use cases |
| **Extendable** | Components can be overridden to build new research systems |
| **Framework-agnostic** | Works with PyTorch, TensorFlow, JAX, scikit-learn, XGBoost, Hugging Face, and more |
| **Understandable** | Written with maintainability and community contribution in mind |

### Use Cases

- **Federated Learning** — Train ML models across distributed nodes without sharing raw data
- **Federated Analytics** — Aggregate statistical insights from distributed datasets
- **Federated Evaluation** — Evaluate models on distributed data
- **On-device AI** — Run and fine-tune models locally on iOS, Android, or browser (via Flower Intelligence)

---

## 2. Repository Structure

```
flower/
├── framework/                  # Core Python framework
│   ├── py/flwr/                # Main Python package
│   │   ├── app/                # Public App APIs
│   │   ├── cli/                # Typer-based CLI
│   │   ├── client/             # Client transport implementations
│   │   ├── clientapp/          # ClientApp abstraction
│   │   ├── common/             # Shared types, context, messages, records
│   │   ├── compat/             # Backwards-compatibility shims
│   │   ├── proto/              # Generated Protocol Buffer Python files
│   │   ├── server/             # ServerApp, strategies, grid, history
│   │   ├── serverapp/          # ServerApp abstraction
│   │   ├── simulation/         # Ray-based simulation engine
│   │   ├── supercore/          # Database state, auth, credential store
│   │   ├── superlink/          # Federation coordination servicer
│   │   └── supernode/          # Client node runtime
│   ├── proto/                  # Protocol Buffer source definitions (.proto)
│   ├── dev/                    # Migration and versioning utilities
│   ├── docs/                   # Sphinx documentation source
│   ├── e2e/                    # End-to-end test projects
│   └── pyproject.toml          # Package metadata and dependencies
│
├── datasets/                   # Federated Datasets package (0.6.0)
│   ├── flwr_datasets/          # Dataset partitioning and visualization
│   └── pyproject.toml
│
├── baselines/                  # 30+ federated learning baseline implementations
│   ├── flwr_baselines/
│   ├── fedavg/
│   ├── fednova/
│   └── [28+ more baselines]
│
├── intelligence/               # On-device AI SDKs
│   ├── swift/                  # iOS / macOS SDK (MLX + transformers)
│   ├── ts/                     # TypeScript / JavaScript SDK
│   └── kt/                     # Kotlin / Android SDK
│
├── examples/                   # 40+ example Flower projects
│   ├── quickstart-pytorch/
│   ├── quickstart-tensorflow/
│   ├── advanced-pytorch/
│   ├── vertical-fl/
│   ├── flowertune-llm/
│   └── [many more]
│
├── dev/                        # Developer tooling (devtool package)
├── benchmarks/                 # Performance benchmarks
├── hub/                        # Hub service
├── .github/                    # CI/CD workflows and templates
├── .devcontainer/              # Dev container configuration
├── Package.swift               # Swift package definition
└── pyproject.toml              # Root project configuration
```

---

## 3. Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flower Federation                        │
│                                                                 │
│   ┌──────────────┐        ┌──────────────────────────────┐     │
│   │  SuperLink   │◄──────►│         SuperCore            │     │
│   │ (Coordinator)│        │  (State DB + Auth + Creds)   │     │
│   └──────┬───────┘        └──────────────────────────────┘     │
│          │ gRPC                                                 │
│    ┌─────┴──────┐                                               │
│    │            │                                               │
│  ┌─┴──────┐  ┌──┴─────┐                                        │
│  │SuperNode│  │SuperNode│  ... (N nodes)                        │
│  │(Client) │  │(Client) │                                       │
│  └─────────┘  └─────────┘                                       │
│                                                                 │
│   ┌──────────────────────────────────────────┐                  │
│   │          SuperExec (Task Executor)       │                  │
│   │   ┌──────────────┐  ┌────────────────┐  │                  │
│   │   │  ServerApp   │  │   ClientApp    │  │                  │
│   │   │  (Strategy)  │  │ (ML Training)  │  │                  │
│   │   └──────────────┘  └────────────────┘  │                  │
│   └──────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | Binary | Role |
|-----------|--------|------|
| **SuperLink** | `flower-superlink` | Federation coordinator; routes tasks between nodes |
| **SuperNode** | `flower-supernode` | Client node; executes ClientApp tasks, reports results |
| **SuperExec** | `flower-superexec` | Launches and manages ServerApp and ClientApp processes |
| **SuperCore** | (embedded in SuperLink) | Persistent state, auth, credential store |
| **ServerApp** | `flwr-serverapp` | Contains the FL strategy and orchestration logic |
| **ClientApp** | `flwr-clientapp` | Contains client-side ML training/evaluation logic |
| **AgentApp** | `flwr-agentapp` | Agent-based execution (agentic AI workflows) |
| **Simulation Engine** | `flwr-simulation` | Ray-based local simulation of federated training |

### Communication Flow

```
ServerApp                    SuperLink                   SuperNode(s)
    │                            │                           │
    │── GetRun ─────────────────►│                           │
    │◄─ RunInfo ─────────────────│                           │
    │                            │◄── RegisterNode ──────────│
    │── CreateTask(train) ───────►│                           │
    │                            │── PushTaskIns ───────────►│
    │                            │◄─ PullTaskRes ────────────│
    │◄── PullTaskRes ────────────│                           │
    │  (aggregates results)       │                           │
    │── CreateTask(evaluate) ────►│                           │
    │         ...                 │          ...              │
```

---

## 4. Core Components

### 4.1 Python Framework (`flwr`)

**Package**: `flwr`  
**Version**: 1.30.0  
**Python**: 3.10 – 3.13  
**Source**: `framework/py/flwr/`

#### 4.1.1 `flwr.app` — Public APIs

Entry point for user-facing types:

| Symbol | Description |
|--------|-------------|
| `UserConfig` | `dict[str, bool | bytes | float | int | str]` — typed key/value config |
| `Metadata` | Message metadata (run_id, message_id, TTL, group_id, message_type) |
| `MessageType` | Constants: `TRAIN`, `EVALUATE`, `QUERY` |
| `Error` | Structured error with code and reason |

#### 4.1.2 `flwr.common` — Shared Primitives

| Module | Key Types |
|--------|-----------|
| `context` | `Context(run_id, node_id, node_config, state, run_config)` |
| `message` | `Message`, `Metadata` |
| `record` | `RecordDict`, `Array`, `ConfigRecord`, `MetricRecord`, `ParametersRecord` |
| `config_record` | Config key/value records for structured configuration |
| `metric_record` | Metric reporting from nodes |
| `parameters_record` | Model weight tensors |
| `secure_aggregation` | Secure aggregation utilities |
| `event_log_plugin` | Event logging hooks |

**`Context` dataclass** — passed to every `ClientApp` and `ServerApp` function:

```python
@dataclass
class Context:
    run_id: int          # Identifies the current run
    node_id: int         # Identifies this node
    node_config: UserConfig  # Persistent per-node config (across runs)
    state: RecordDict    # Local per-run scratchpad (never leaves the node)
    run_config: UserConfig   # Per-run configuration
```

#### 4.1.3 `flwr.client` — Client Transport

Implements the client side of the communication protocol.

| Module | Description |
|--------|-------------|
| `grpc_adapter_client.py` | gRPC adapter client (primary) |
| `grpc_rere_client.py` | Reliable-reconnect gRPC client |
| `rest_client.py` | REST/HTTP client (optional) |
| `message_handler/` | Dispatches incoming messages to handler functions |
| `mod/` | Client-side middleware chain |

#### 4.1.4 `flwr.clientapp` — ClientApp Abstraction

```python
from flwr.client import ClientApp, NumPyClient

app = ClientApp()

@app.train()
def train(msg: Message, ctx: Context) -> Message:
    # Access model parameters from msg.content
    # Train locally
    # Return updated parameters
    ...

@app.evaluate()
def evaluate(msg: Message, ctx: Context) -> Message:
    ...
```

**`ClientApp` decorators**: `@app.train()`, `@app.evaluate()`, `@app.query()`

#### 4.1.5 `flwr.server` — Server & Strategies

**`ServerApp`** wraps the FL orchestration logic:

```python
from flwr.server import ServerApp, ServerConfig

app = ServerApp()

@app.main()
def main(grid: Grid, ctx: Context) -> None:
    # Use grid to sample clients and run rounds
    ...
```

**Strategy interface** (`flwr.server.strategy.Strategy`):

```python
class Strategy(ABC):
    def initialize_parameters(self, client_manager) -> Optional[Parameters]
    def configure_fit(self, server_round, parameters, client_manager) -> List[Tuple[ClientProxy, FitIns]]
    def aggregate_fit(self, server_round, results, failures) -> Tuple[Optional[Parameters], Dict]
    def configure_evaluate(self, server_round, parameters, client_manager) -> List[Tuple[ClientProxy, EvaluateIns]]
    def aggregate_evaluate(self, server_round, results, failures) -> Tuple[Optional[float], Dict]
    def evaluate(self, server_round, parameters) -> Optional[Tuple[float, Dict]]
```

**`Grid`** / **`Driver`** — abstractions for sampling clients and dispatching work:

| Method | Description |
|--------|-------------|
| `get_node_ids()` | Returns IDs of available nodes |
| `send_and_receive(messages)` | Broadcasts messages and collects replies |

**`ClientManager`** — manages client lifecycle and sampling:
- `register(client)` / `unregister(client)` 
- `sample(num_clients, min_num_clients, criterion)` → `List[ClientProxy]`

**`History`** — accumulates metrics across rounds:
- `add_loss_distributed(server_round, loss)`
- `add_metrics_distributed(server_round, metrics)`
- `losses_distributed`, `metrics_distributed`, `losses_centralized`, `metrics_centralized`

#### 4.1.6 `flwr.simulation` — Ray-based Simulation

Enables local multi-client simulation without deploying distributed infrastructure:

```python
from flwr.simulation import run_simulation

run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=10,
    backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
)
```

**Dependencies**: `ray==2.51.1` (Python 3.10–3.13; Linux/macOS only on 3.13)

#### 4.1.7 `flwr.supercore` — Core State Management

Manages persistent state for the federation server:

| Sub-package | Description |
|-------------|-------------|
| `state/` | SQLAlchemy models + Alembic migrations for CoreState and LinkState |
| `credential_store/` | Stores and retrieves node/user credentials |
| `auth/` | Authentication logic |
| `license_plugin/` | License enforcement |
| `corestate/` | Task and run state management |
| `inflatable/` | Object serialization (inflate/deflate) |
| `interceptors/` | gRPC interceptors for auth and logging |
| `primitives/` | Low-level data structures |

#### 4.1.8 `flwr.superlink` — Federation Coordinator

Routes tasks and messages between ServerApp and SuperNodes:

| Sub-package | Description |
|-------------|-------------|
| `servicer/` | gRPC servicer implementations for fleet and exec protocols |
| `federation/` | Federation lifecycle management |
| `artifact_provider/` | FAB (Federated App Bundle) delivery to nodes |
| `auth_plugin/` | Pluggable authentication backends |

#### 4.1.9 `flwr.supernode` — Client Node Runtime

Runs on each participant's machine:

| Sub-package | Description |
|-------------|-------------|
| `cli/flower_supernode.py` | Entry point for `flower-supernode` command |
| `nodestate/` | Local node state persistence |
| `runtime/` | Task execution loop |
| `servicer/` | gRPC service for receiving tasks |

---

### 4.2 Federated Datasets (`flwr-datasets`)

**Package**: `flwr-datasets`  
**Version**: 0.6.0  
**Source**: `datasets/flwr_datasets/`

Provides utilities for creating non-IID federated dataset partitions from Hugging Face datasets.

**Key Features**:
- Partition strategies: IID, Dirichlet (heterogeneous), shard-based, pathological
- Visualization tools for data distribution analysis
- Built on top of `datasets>=4.0.0` (Hugging Face)
- Optional vision support: `pillow>=12.1.1`
- Optional audio support: `torchcodec>=0.7.0`

**Core Classes**:

```python
from flwr_datasets import FederatedDataset

fds = FederatedDataset(dataset="cifar10", partitioners={"train": 100})
partition = fds.load_partition(idx=0, split="train")
```

---

### 4.3 Flower Intelligence (Multi-platform SDKs)

Flower Intelligence enables on-device AI inference and fine-tuning with optional remote compute fallback.

#### TypeScript / JavaScript SDK

**Source**: `intelligence/ts/`  
**NPM Package**: `@flwr/flwr-intelligence` (or similar)

**Dependencies**:

| Package | Version | Purpose |
|---------|---------|---------|
| `@huggingface/transformers` | `>=3.8.1` | Model inference via ONNX/WebGPU |
| `@mlc-ai/web-llm` | `>=0.2.81` | Browser-native LLM execution |
| `get-random-values` | `>=3.0.0` | Cryptographic randomness |
| `tar` | `>=7.5.9` | Model archive handling |

**Examples**: `intelligence/ts/examples/`
- Web inference
- Streaming text generation
- Encrypted model loading

#### Swift SDK (iOS / macOS)

**Source**: `intelligence/swift/`  
**Package**: `FlowerIntelligence` (Swift Package Manager)

**Dependencies**:

| Package | Purpose |
|---------|---------|
| `ml-explore/mlx-swift>=0.12.1` | Apple Silicon ML acceleration |
| `huggingface/swift-transformers>=1.0.0` | Transformer model loading |
| `ml-explore/mlx-swift-lm>=2.29.1` | LLM utilities on MLX |
| `swift-async-algorithms>=1.0.0` | Async stream handling |
| `swift-crypto>=3.10.1` | Cryptographic operations |

**Platform Targets**: iOS, macOS

#### Kotlin / Android SDK

**Source**: `intelligence/kt/`  
**Build System**: Gradle (`build.gradle.kts`)

---

### 4.4 Flower Baselines

**Source**: `baselines/`  
**Version**: 1.0.0

A curated collection of 30+ Flower implementations that reproduce published federated learning research papers.

**Core Dependencies**:
- `flwr[simulation]>=1.3.0`
- `torch==2.4.1`, `torchvision==0.19.1`
- `hydra-core>=1.2.0` (config management)
- `scikit-learn>=1.2.1`

**Available Baselines** (partial list):

| Baseline | Paper Reference |
|----------|----------------|
| `fedavg` | McMahan et al., 2017 — Communication-Efficient Learning of Deep Networks |
| `fednova` | Wang et al., 2020 — Tackling the Objective Inconsistency Problem |
| `fedbn` | Li et al., 2021 — Federated Learning on Non-IID Features via Local Batch Norm |
| `fedprox` | Li et al., 2020 — Federated Optimization in Heterogeneous Networks |
| `moon` | Li et al., 2021 — Model-Contrastive Federated Learning |
| `dasha` | Tyurin & Richtárik, 2022 — DASHA: Distributed Nonconvex Optimization |
| `depthfl` | DepthFL paper implementation |
| `heterofl` | HeteroFL — Computation and Communication Efficient Federated Learning |
| `fedsimclr` | SimCLR-based federated learning |
| `fedvssl` | Federated Video Self-Supervised Learning |
| `flowertune-llm` | LLM fine-tuning via FL |

---

## 5. Data Models & Protocols

### Protocol Buffer Definitions

All inter-component communication is defined in `.proto` files under `framework/proto/`:

| Proto File | Description |
|------------|-------------|
| `transport.proto` | Core transport abstractions |
| `task.proto` | Task definitions and lifecycle |
| `message.proto` | Client/server message envelope |
| `node.proto` | Node registration and metadata |
| `fleet.proto` | Fleet coordination (SuperLink ↔ SuperNode) |
| `control.proto` | Control plane messages |
| `clientappio.proto` | ClientApp I/O protocol |
| `serverappio.proto` | ServerApp I/O protocol |
| `heartbeat.proto` | Node heartbeat protocol |
| `federation.proto` | Federation management |
| `federation_config.proto` | Federation configuration |
| `fab.proto` | Federated App Bundle format |
| `error.proto` | Structured error types |
| `log.proto` | Centralized log aggregation |

### Message Structure

```
Message
├── metadata: Metadata
│   ├── run_id: int
│   ├── message_id: str (UUID)
│   ├── src_node_id: int
│   ├── dst_node_id: int
│   ├── reply_to_message_id: str
│   ├── group_id: str
│   ├── ttl: float (seconds)
│   └── message_type: str  ("train.fit" | "evaluate.evaluate" | ...)
└── content: RecordDict
    ├── ParametersRecord  (model weights as Arrays)
    ├── MetricRecord      (loss, accuracy, etc.)
    └── ConfigRecord      (hyperparameters, settings)
```

### Message Types

| Category | Action | Full Type |
|----------|--------|-----------|
| `TRAIN` | `fit` | `"train.fit"` |
| `EVALUATE` | `evaluate` | `"evaluate.evaluate"` |
| `QUERY` | `get_properties` | `"query.get_properties"` |

### Federated App Bundle (FAB)

A FAB is a self-contained archive packaging the ServerApp and ClientApp code for distribution to SuperNodes:

```
app.fab
├── pyproject.toml      (app metadata and dependencies)
├── server_app.py       (ServerApp entry point)
├── client_app.py       (ClientApp entry point)
└── [additional modules]
```

---

## 6. Federated Learning Strategies

All strategies are in `framework/py/flwr/server/strategy/` and implement the abstract `Strategy` base class.

### Standard Strategies

| Strategy | Class | Description |
|----------|-------|-------------|
| **FedAvg** | `FedAvg` | Federated Averaging — baseline FL algorithm |
| **FedAvgM** | `FedAvgM` | FedAvg with server-side momentum |
| **FedOpt** | `FedOpt` | Federated Optimization (server-side optimizer) |
| **FedAdam** | `FedAdam` | FedOpt with Adam server optimizer |
| **FedAdagrad** | `FedAdagrad` | FedOpt with Adagrad server optimizer |
| **FedYogi** | `FedYogi` | FedOpt with Yogi server optimizer |
| **FedProx** | `FedProx` | FedAvg with proximal term for heterogeneous systems |
| **FedMedian** | `FedMedian` | Robust aggregation via coordinate-wise median |
| **FedTrimmedAvg** | `FedTrimmedAvg` | Trimmed mean aggregation |
| **QFedAvg** | `QFedAvg` | Communication-efficient FL with quantization |
| **FedAvgAndroid** | `FedAvgAndroid` | FedAvg optimized for Android clients |

### Byzantine-Robust Strategies

| Strategy | Class | Description |
|----------|-------|-------------|
| **Krum** | `Krum` | Multi-Krum Byzantine-robust aggregation |
| **Bulyan** | `Bulyan` | Bulyan Byzantine-robust aggregation |
| **FaultTolerantFedAvg** | `FaultTolerantFedAvg` | FedAvg tolerant of client failures |

### Differential Privacy Strategies

| Strategy | Class | Description |
|----------|-------|-------------|
| **DPFedAvgFixed** | `DPFedAvgFixed` | DP-FedAvg with fixed clipping threshold |
| **DPFedAvgAdaptive** | `DPFedAvgAdaptive` | DP-FedAvg with adaptive clipping |
| **DP Client-Side Fixed Clipping** | `DifferentialPrivacyClientSideFixedClipping` | Client-side gradient clipping (fixed) |
| **DP Client-Side Adaptive Clipping** | `DifferentialPrivacyClientSideAdaptiveClipping` | Client-side gradient clipping (adaptive) |
| **DP Server-Side Fixed Clipping** | `DifferentialPrivacyServerSideFixedClipping` | Server-side noise injection (fixed) |
| **DP Server-Side Adaptive Clipping** | `DifferentialPrivacyServerSideAdaptiveClipping` | Server-side noise injection (adaptive) |

### Tree-based Strategies (XGBoost)

| Strategy | Class | Description |
|----------|-------|-------------|
| **FedXgbBagging** | `FedXgbBagging` | XGBoost federation via bagging |
| **FedXgbCyclic** | `FedXgbCyclic` | XGBoost federation with cyclic training |
| **FedXgbNnAvg** | `FedXgbNnAvg` | XGBoost + neural network averaging |

### Implementing a Custom Strategy

```python
from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes

class MyStrategy(Strategy):
    def initialize_parameters(self, client_manager):
        return None  # Random initialization

    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(10, min_num_clients=5)
        config = {"learning_rate": 0.01, "epochs": 1}
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        # Custom aggregation logic
        ...

    def configure_evaluate(self, server_round, parameters, client_manager):
        ...

    def aggregate_evaluate(self, server_round, results, failures):
        ...

    def evaluate(self, server_round, parameters):
        return None  # No centralized evaluation
```

---

## 7. CLI Reference

**Entry point**: `flwr` (installed via `pip install flwr`)

### Top-Level Commands

```
flwr
├── new          Create a new Flower project from a template
├── run          Run a Flower app (local or remote federation)
├── build        Build a Docker image for the app
├── ls / list    List apps or active runs
├── stop         Stop a running app
├── login        Authenticate with a Flower federation
├── log          Stream or view execution logs
├── pull         Pull run artifacts
├── install      Install FAB dependencies
├── supernode    Manage SuperNodes
│   ├── register
│   ├── unregister
│   └── list
├── federation   Manage federations
│   ├── create
│   ├── ls
│   ├── remove_account
│   ├── add_supernode
│   ├── remove_supernode
│   ├── archive
│   └── invite
│       ├── create
│       ├── list
│       ├── accept
│       ├── reject
│       └── revoke
└── app          Manage published apps
    ├── review
    └── publish
```

### Running a Federated App Locally (Simulation)

```bash
# Create a new project
flwr new my-fl-app --framework pytorch --username myorg

# Run in simulation mode
cd my-fl-app
flwr run .

# Run against a deployed federation
flwr run . my-federation
```

### Deploying Infrastructure

```bash
# Start SuperLink (federation coordinator)
flower-superlink --insecure

# Start SuperNode (client node)
flower-supernode --insecure \
  --superlink localhost:9091 \
  --clientappio-api-address 0.0.0.0:9094

# Start SuperExec (task executor)
flower-superexec --insecure
```

---

## 8. Configuration

### App Configuration (`pyproject.toml`)

Every Flower app has a `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-fl-app"
version = "1.0.0"
dependencies = ["flwr>=1.30.0", "torch>=2.0"]

[tool.flwr.app]
publisher = "myorg"

[tool.flwr.app.components]
serverapp = "my_fl_app.server:app"
clientapp = "my_fl_app.client:app"

[tool.flwr.app.config]
num-server-rounds = 3
fraction-fit = 0.5

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 2
```

### Run Configuration via `Context`

Runtime configuration is injected through `Context.run_config`:

```python
@app.train()
def train(msg: Message, ctx: Context) -> Message:
    lr = ctx.run_config.get("learning_rate", 0.01)
    epochs = int(ctx.run_config.get("local-epochs", 1))
    ...
```

### Framework `pyproject.toml` Tool Settings

```toml
[tool.pytest.ini_options]
addopts = "--strict-markers"

[tool.mypy]
strict = true

[tool.black]
line-length = 88

[tool.isort]
profile = "black"

[tool.ruff]
line-length = 88
```

---

## 9. Database Schema

Flower uses SQLAlchemy with Alembic for persistent state management. There are three logical databases:

### CoreState Database (Task Execution)

| Table | Primary Key | Key Columns | Purpose |
|-------|-------------|-------------|---------|
| `nonce_store` | `nonce_id` | `value`, `expiry` | One-time nonces for replay protection |
| `fab` | `fab_id` | `hash`, `content` (bytes) | Federated App Bundle storage |
| `task` | `task_id` (UUID) | `type`, `run_id`, `status`, `created_at`, `ttl` | Task lifecycle tracking |
| `task_message` | `message_id` (UUID) | `task_id`, `src_node_id`, `dst_node_id`, `content` | Inter-task message routing |
| `task_logs` | `log_id` | `task_id`, `timestamp`, `level`, `message` | Task execution logs |

### LinkState Database (Federation State)

| Table | Primary Key | Key Columns | Purpose |
|-------|-------------|-------------|---------|
| `node` | `node_id` (int64) | `owner`, `status`, `last_heartbeat`, `public_key` | Node registry |
| `run` | `run_id` (int64) | `fab_hash`, `override_config`, `status`, `metrics` | FL run lifecycle |
| `logs` | `log_id` | `run_id`, `node_id`, `timestamp`, `content` | Aggregated run logs |

### ObjStore Database (Object Storage)

Binary/tensor object storage for large model weights and artifacts.

### Migrations

Database migrations use Alembic and are located at `framework/py/flwr/supercore/state/alembic/`.

```bash
# Generate a new migration (from framework/ directory)
python -m dev.generate_migration "Add nonce_store table"

# Apply migrations
alembic -c framework/py/flwr/supercore/state/alembic.ini upgrade head
```

---

## 10. Transport Layer

### gRPC (Primary)

Flower uses gRPC as its primary transport protocol:

- **Port 9091** — SuperLink fleet API (SuperNode ↔ SuperLink)
- **Port 9092** — SuperLink exec API (SuperExec ↔ SuperLink)
- **Port 9093** — SuperLink federation management API
- **Port 9094** — SuperNode ClientApp I/O API
- **Port 9095** — SuperNode ServerApp I/O API

**Interceptors** (applied at gRPC level):
- Authentication interceptor (verifies tokens/certificates)
- Logging interceptor
- Rate limiting interceptor

### REST (Optional)

A REST transport is available for environments where gRPC is blocked:

```toml
[project.optional-dependencies]
rest = ["starlette>=0.50.0,<0.51.0", "uvicorn[standard]>=0.40.0,<0.41.0"]
```

Enable REST on the SuperNode:
```bash
flower-supernode --rest --insecure --superlink localhost:9091
```

### Reliable Reconnect

The `grpc_rere_client.py` implements automatic reconnection with backoff for production deployments where transient network failures are expected.

---

## 11. Security

### Transport Security (TLS)

All production deployments should use TLS:

```bash
# SuperLink with TLS
flower-superlink \
  --ssl-ca-certfile ca.crt \
  --ssl-certfile server.crt \
  --ssl-keyfile server.key

# SuperNode with TLS
flower-supernode \
  --ssl-ca-certfile ca.crt \
  --superlink localhost:9091
```

### Authentication

Flower supports pluggable authentication via `flwr.superlink.auth_plugin`:

- **Certificate-based auth** — mTLS with client certificates
- **Token-based auth** — Bearer token authentication
- **Public-key auth** — Node registration with public keys

Node public keys are stored in the `node` table (`public_key` column).

### Nonce-based Replay Protection

The `nonce_store` table stores one-time nonces to prevent replay attacks. Each authenticated request consumes a nonce with an expiry timestamp.

### Secure Aggregation

`flwr.common.secure_aggregation` provides secure aggregation protocols that prevent the server from inspecting individual client updates:

- Masks individual model updates with random noise
- Noise cancels when aggregated, preserving the sum
- Requires a minimum threshold of clients to proceed

### Differential Privacy

Server-side and client-side DP wrappers are available as strategy decorators (see [Section 6](#6-federated-learning-strategies)):

- **Fixed clipping**: Clip gradient norms at a fixed threshold, then add calibrated Gaussian noise
- **Adaptive clipping**: Automatically estimates the optimal clipping threshold per round

---

## 12. Testing

### Test Organization

```
framework/py/flwr/
├── app/                    app_test.py
├── cli/                    *_test.py per module
├── client/                 *_test.py per module
├── common/                 *_test.py per module
├── server/                 *_test.py, strategy/*_test.py
├── simulation/             *_test.py
└── supercore/              *_test.py

framework/e2e/
├── e2e-bare/               Minimal setup
├── e2e-bare-auth/          Authentication
├── e2e-bare-https/         HTTPS/TLS
├── e2e-pytorch/            PyTorch integration
├── e2e-tensorflow/         TensorFlow integration
├── e2e-pandas/             Pandas/analytics
├── e2e-serverapp-heartbeat/ Heartbeat testing
└── [more framework-specific e2e tests]

datasets/flwr_datasets/     *_test.py per module
baselines/flwr_baselines/   *_test.py per module
```

### Running Tests

```bash
# Unit tests (from framework/)
python -m pytest framework/py/flwr/ -qq

# With type checking
mypy --strict framework/py/flwr/

# Linting
ruff check framework/py/flwr/
black --check framework/py/flwr/
isort --check-only framework/py/flwr/

# End-to-end tests
cd framework/e2e/e2e-pytorch
flwr run .
```

### Code Quality Standards

| Tool | Standard |
|------|----------|
| **Black** | Line length 88 |
| **isort** | `profile = "black"` |
| **MyPy** | `strict = true` |
| **Pylint** | Default config |
| **Ruff** | Line length 88 |

---

## 13. Build & Deployment

### Python Package Build

Flower uses [Poetry](https://python-poetry.org/) with [UV](https://github.com/astral-sh/uv):

```bash
# Install dependencies
cd framework/
uv sync

# Build the package
poetry build

# Publish to PyPI
twine upload dist/*
```

### Protocol Buffer Compilation

Proto files are compiled using the `devtool` package:

```bash
cd framework/
python -m dev.compile_protos
```

Output: Python files in `framework/py/flwr/proto/`  
Output: MyPy stubs in `framework/py/flwr/proto/`

### Docker

A dev container is configured at `.devcontainer/`:

```json
// .devcontainer/devcontainer.json
{
  "name": "Flower Dev Container",
  "build": { "dockerfile": "Dockerfile" },
  "features": { "ghcr.io/devcontainers/features/python:1": { "version": "3.11" } }
}
```

Build a production Docker image for your app:

```bash
flwr build .
docker push my-registry/my-fl-app:latest
```

### CI/CD (GitHub Actions)

| Workflow | Trigger | What it Tests |
|----------|---------|---------------|
| `framework-test.yml` | push/PR | Python 3.10–3.13 unit tests, mypy, lint |
| `framework-e2e.yml` | push/PR | End-to-end federated learning runs |
| `framework-release.yml` | tag push | PyPI publishing |
| `framework-docs.yml` | push | Sphinx documentation build |
| `datasets-test.yml` | push/PR | flwr-datasets tests |
| `baselines.yml` | push/PR | Baseline project tests |
| `intelligence-ts.yml` | push/PR | TypeScript SDK build and test |
| `intelligence-swift.yml` | push/PR | Swift SDK build and test |

### Version Management

The canonical version is stored in `flwr.supercore.version` and in `framework/pyproject.toml`.

```bash
# Update version across all files
python dev/update_version.py 1.31.0
```

---

## 14. Development Guide

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/flwrlabs/flower.git
cd flower

# Install Python framework in editable mode
cd framework/
uv sync
uv pip install -e ".[simulation,rest]"

# Set up pre-commit hooks
pre-commit install
```

### Project Structure for a New Flower App

```bash
flwr new my-app --framework pytorch --username myorg
cd my-app/
```

Generated structure:
```
my-app/
├── pyproject.toml          # App config and dependencies
├── README.md
└── my_app/
    ├── __init__.py
    ├── client_app.py       # ClientApp definition
    ├── server_app.py       # ServerApp definition
    └── task.py             # ML training functions
```

### Writing a ClientApp

```python
# client_app.py
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

app = ClientApp()

class FlowerClient(NumPyClient):
    def fit(self, parameters, config):
        # Load parameters into model
        # Train locally for config["local-epochs"] epochs
        # Return updated parameters, num_examples, metrics
        ...

    def evaluate(self, parameters, config):
        # Load parameters into model
        # Evaluate on local test set
        # Return loss, num_examples, metrics
        ...

@app.client()
def client_fn(context: Context):
    return FlowerClient().to_client()
```

### Writing a ServerApp

```python
# server_app.py
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

app = ServerApp()

@app.main()
def main(grid, context: Context):
    strategy = FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.25,
        min_fit_clients=2,
        min_available_clients=2,
    )
    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))
    
    # Run using the classic API
    from flwr.server import run_via_grid
    run_via_grid(grid, strategy, config)
```

### Database Migrations

When modifying SQLAlchemy models in `supercore/state/schema/`:

```bash
cd framework/
python -m dev.generate_migration "Describe your schema change"
# Review the generated migration in supercore/state/alembic/versions/
# Apply: alembic upgrade head
```

### Adding a New Strategy

1. Create `framework/py/flwr/server/strategy/my_strategy.py`
2. Inherit from `Strategy` and implement all abstract methods
3. Export from `framework/py/flwr/server/strategy/__init__.py`
4. Add tests at `framework/py/flwr/server/strategy/my_strategy_test.py`

---

## 15. Examples & Baselines

### Quickstart Examples

| Example | Framework | Path |
|---------|-----------|------|
| PyTorch | PyTorch | `examples/quickstart-pytorch/` |
| TensorFlow | TensorFlow | `examples/quickstart-tensorflow/` |
| Hugging Face | Transformers | `examples/quickstart-huggingface/` |
| JAX | JAX | `examples/quickstart-jax/` |
| scikit-learn | scikit-learn | `examples/quickstart-sklearn/` |
| XGBoost | XGBoost | `examples/quickstart-xgboost/` |
| PyTorch Lightning | Lightning | `examples/quickstart-pytorch-lightning/` |
| fastai | fastai | `examples/quickstart-fastai/` |
| Pandas | Pandas | `examples/quickstart-pandas/` |
| Android (TFLite) | TFLite | `examples/android/` |
| iOS (CoreML) | CoreML | `examples/ios/` |

### Advanced Examples

| Example | Description | Path |
|---------|-------------|------|
| `advanced-pytorch` | Multi-GPU, custom mods, checkpointing | `examples/advanced-pytorch/` |
| `vertical-fl` | Vertical federated learning | `examples/vertical-fl/` |
| `flowertune-llm` | LLM fine-tuning via federated learning | `examples/flowertune-llm/` |
| `custom-mods` | Writing and chaining ClientApp mods | `examples/custom-mods/` |
| `df-analytics` | Federated analytics with DataFrames | `examples/df-analytics/` |
| `whisper` | Federated fine-tuning of OpenAI Whisper | `examples/whisper/` |

### Running an Example

```bash
cd examples/quickstart-pytorch/
pip install -e .
flwr run .
```

### Baselines

Each baseline in `baselines/` follows the same structure:

```
baselines/fedavg/
├── pyproject.toml
├── README.md
└── fedavg/
    ├── main.py            # Hydra entry point
    ├── client.py
    ├── server.py
    ├── dataset.py
    ├── model.py
    └── conf/
        └── base.yaml      # Hydra config
```

```bash
cd baselines/fedavg/
pip install -e .
python -m fedavg.main
```

---

## Appendix A: Dependency Summary

### Python Framework (`flwr`)

| Dependency | Version Range | Purpose |
|------------|---------------|---------|
| `numpy` | `>=1.26.0,<3.0.0` | Numerical arrays for model parameters |
| `grpcio` | `>=1.70.0,<2.0.0` | gRPC communication |
| `grpcio-health-checking` | `>=1.70.0,<2.0.0` | gRPC health check protocol |
| `protobuf` | `>=5.28.0,<7.0.0` | Protocol Buffer serialization |
| `cryptography` | `>=46.0.5,<47.0.0` | TLS and cryptographic operations |
| `pycryptodome` | `>=3.18.0,<4.0.0` | Additional cryptographic primitives |
| `iterators` | `>=0.0.2,<0.0.3` | Iterator utilities |
| `typer` | `>=0.13.0,<0.21.0` | CLI framework |
| `uv` | `>=0.11.3,<0.12.0` | Package installer (used by `flwr run`) |
| `tomli` / `tomli-w` | `>=2.0.1` / `>=1.0.0` | TOML parsing and writing |
| `pathspec` | `>=0.12.1,<0.13.0` | `.gitignore`-style path matching |
| `rich` | `>=13.5.0,<14.0.0` | Terminal formatting |
| `pyyaml` | `>=6.0.2,<7.0.0` | YAML configuration |
| `requests` | `>=2.31.0,<3.0.0` | HTTP client |
| `click` | `>=8.0.0,<9.0.0` | CLI building (used by Typer) |
| `packaging` | `>=24.0,<26.0` | Version parsing |
| `SQLAlchemy` | `>=2.0.45,<3.0.0` | ORM for persistent state |
| `alembic` | `>=1.18.1,<2.0.0` | Database migrations |

**Optional**:

| Extra | Dependency | Version |
|-------|------------|---------|
| `simulation` | `ray` | `==2.51.1` |
| `rest` | `starlette` | `>=0.50.0,<0.51.0` |
| `rest` | `uvicorn[standard]` | `>=0.40.0,<0.41.0` |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Federation** | A group of SuperNodes coordinated by a SuperLink |
| **SuperLink** | The central federation coordinator |
| **SuperNode** | A client-side node that executes ClientApp tasks |
| **SuperExec** | Task executor that manages ServerApp and ClientApp processes |
| **SuperCore** | Embedded state management layer within SuperLink |
| **ClientApp** | User code defining local training/evaluation logic |
| **ServerApp** | User code defining the FL strategy and orchestration |
| **FAB** | Federated App Bundle — packaged app distributed to nodes |
| **Context** | Runtime object holding run_id, node_id, config, and local state |
| **RecordDict** | Typed dictionary holding model parameters, metrics, and configs |
| **Strategy** | Pluggable aggregation algorithm (FedAvg, FedAdam, etc.) |
| **Round** | One iteration of: select clients → train → aggregate → evaluate |
| **Grid** | Server-side abstraction for sampling and messaging nodes |
| **Mod** | Middleware that wraps a ClientApp or ServerApp function |
| **IID** | Independent and identically distributed (data distribution) |
| **Non-IID** | Heterogeneous data distribution across nodes |
| **DP** | Differential Privacy — formal privacy guarantee |
| **Secure Aggregation** | Cryptographic protocol hiding individual updates from the server |

---

*Documentation generated from codebase analysis. Last updated: 2026-05-20.*
