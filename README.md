# FedRoute: Privacy-Preserving Federated Learning for Internet of Vehicles

<div align="center">

**A Privacy-Preserving Federated Multi-Task Learning Framework for Dual Recommendations in Internet of Vehicles**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SUMO](https://img.shields.io/badge/SUMO-1.15+-green.svg)](https://sumo.dlr.de/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Demo](#-demo)
- [Experiments](#-experiments)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Research Contributions](#-research-contributions)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

**FedRoute** addresses the challenge of providing personalized recommendations in Internet of Vehicles (IoV) environments while preserving user privacy. The framework implements a novel **Federated Multi-Task Learning (FMTL)** approach that simultaneously handles:

- 🗺️ **Point-of-Interest (POI) Recommendations**: Intelligent path planning and destination suggestions
- 🎵 **Music Recommendations**: Context-aware music suggestions based on driving conditions

### Why FedRoute?

Traditional centralized recommendation systems require collecting sensitive user data (location history, music preferences) on central servers, raising significant privacy concerns. FedRoute solves this by:

✅ **Keeping data on-device**: User data never leaves the vehicle  
✅ **Privacy guarantees**: Differential privacy and secure aggregation  
✅ **Context-aware learning**: Leverages spatial-temporal-mobility contexts  
✅ **Efficient collaboration**: Intelligent client selection based on contextual similarity  

---

## ✨ Key Features

### 🔐 Privacy-Preserving Mechanisms
- **Differential Privacy**: Configurable privacy budgets (ε) with Gaussian noise
- **Secure Aggregation**: Homomorphic encryption for model updates
- **Local Training**: All sensitive data remains on client devices

### 🧠 Advanced Federated Learning
- **Federated Multi-Task Learning (FMTL)**: Shared and task-specific layers
- **Intelligent Client Selection**: Multi-objective selection based on:
  - Contextual similarity (spatial, temporal, mobility)
  - Data quality and quantity
  - Communication costs
- **Adaptive Aggregation**: Weighted FedAvg with client contribution scoring

### 🚗 IoV-Specific Optimizations
- **High-Mobility Support**: Handles frequent disconnections and mobility patterns
- **Heterogeneous Data**: Manages non-IID data distributions across vehicles
- **SUMO Integration**: Realistic traffic simulation for evaluation
- **Real-time Visualization**: Live training dashboard with matplotlib

### 📊 Comprehensive Evaluation
- Baseline comparisons: Centralized, Local-only, Standard FedAvg
- Ablation studies: Component-wise impact analysis
- Privacy-utility trade-offs: Performance vs. privacy guarantees
- Scalability analysis: 10-200 clients

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Federated Server                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Global     │  │   Client     │  │    Secure    │     │
│  │   Model      │  │  Selection   │  │ Aggregation  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          ▲         │
              Model       │         │    Updates
              Broadcast   │         ▼    + Noise
                    ┌─────────────────────────┐
                    │   Selected Vehicles     │
                    │  (Context Similarity)   │
                    └─────────────────────────┘
                          │         │
              ┌───────────┘         └──────────┐
              ▼                                 ▼
    ┌─────────────────┐                ┌─────────────────┐
    │   Vehicle 1     │                │   Vehicle N     │
    │  ┌───────────┐  │                │  ┌───────────┐  │
    │  │Local Model│  │      ...       │  │Local Model│  │
    │  │  Training │  │                │  │  Training │  │
    │  └───────────┘  │                │  └───────────┘  │
    │  Local Data     │                │  Local Data     │
    └─────────────────┘                └─────────────────┘
```

### FMTL Model Architecture

```
Input: [Spatial, Temporal, Mobility Context]
         │
         ▼
┌──────────────────┐
│  Shared Encoder  │ ← Learns common representations
│   (3 layers)     │
└──────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│  POI   │ │ Music  │ ← Task-specific heads
│  Head  │ │  Head  │
└────────┘ └────────┘
    │         │
    ▼         ▼
  POI      Music
 Ranking  Ranking
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- SUMO 1.15+ (for traffic simulation)
- CUDA-capable GPU (optional, for faster training)

### Step 1: Clone the Repository

```bash
git clone git@github.com:Vatsal-Jha256/FedRoute.git
cd FedRoute
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install SUMO (for demo)

**Ubuntu/Debian:**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

**macOS:**
```bash
brew install sumo
```

**Windows:** Download from [SUMO website](https://sumo.dlr.de/docs/Downloads.php)

---

## 🚀 Quick Start

### Run Complete Experiments

```bash
# Run all experiments (baseline, ablation, privacy-utility)
python experiments/comprehensive_experiments.py
```

### Generate Publication Figures

```bash
# Generate all paper figures
python src/utils/generate_publication_figures.py
```

### View Results

Results are saved to:
- `results/experiments/` - CSV files with detailed metrics
- `results/figures/` - PDF/PNG visualizations

---

## 🎮 Demo

### Interactive SUMO Demo with Live Visualization

Run the complete federated learning demo with SUMO traffic simulation and real-time training dashboard:

```bash
cd demo
python run_demo.py
```

**What happens:**
1. 🚗 SUMO GUI opens with 10 vehicles in a city network
2. 📡 Federated learning server starts
3. 🤖 10 vehicle clients connect and register
4. 📊 Live matplotlib dashboard shows:
   - Training progress across 15 rounds
   - Real-time accuracy curves (POI, Music, Combined)
   - Connected clients and selected participants
   - Round-by-round status messages

**Demo Features:**
- ✅ Visual vehicle tracking in SUMO
- ✅ Real-time FL training progress
- ✅ Non-blocking visualization (training continues while viewing)
- ✅ Colored vehicles with IDs for easy tracking
- ✅ Complete 15-round training cycle

**Demo Duration:** ~3-5 minutes for full 15 rounds

---

## 🧪 Experiments

### Available Experiments

#### 1. Baseline Comparison
```bash
python experiments/baseline_experiments.py
```
Compares FedRoute against:
- Centralized learning (upper bound)
- Local-only learning (lower bound)
- Standard FedAvg
- Independent FL (no shared layers)

#### 2. Federated Learning Experiments
```bash
python experiments/federated_learning_experiments.py
```
Comprehensive FL experiments including:
- FedRoute variants (basic, FMTL, with/without privacy)
- Ablation studies (random selection, no privacy)
- Scalability analysis (10-200 clients)

#### 3. Custom Experiments
```bash
python experiments/comprehensive_experiments.py
```
Full experimental suite with all comparisons and analyses.

---

## 📈 Results

### Key Performance Metrics

| Method | POI Accuracy | Music Accuracy | Combined | Privacy | Communication Cost |
|--------|--------------|----------------|----------|---------|-------------------|
| **FedRoute (Full)** | **0.7845** | **0.7923** | **0.7884** | ✅ ε=1.0 | Medium |
| FedRoute (No Privacy) | 0.7891 | 0.8012 | 0.7952 | ❌ None | Medium |
| FedAvg | 0.7523 | 0.7634 | 0.7579 | ❌ None | High |
| Centralized | 0.8234 | 0.8321 | 0.8278 | ❌ None | N/A |
| Local Only | 0.6512 | 0.6734 | 0.6623 | ✅ Perfect | Low |

### Convergence Performance

![Convergence](results/figures/figure1_convergence.png)

*FedRoute achieves near-centralized performance while maintaining strong privacy guarantees.*

### Privacy-Utility Trade-off

![Privacy-Utility](results/figures/figure2_privacy_utility.png)

*Performance degradation vs. privacy budget (ε). FedRoute maintains >95% accuracy even with ε=0.5.*

---

## 📁 Project Structure

```
FedRoute/
├── 📂 src/                          # Source code
│   ├── 📂 federated/                # Federated learning components
│   │   ├── client_selection.py     # Context-aware client selection
│   │   └── privacy.py               # Privacy mechanisms (DP, SecAgg)
│   ├── 📂 models/                   # Model architectures
│   │   └── fmtl_model.py            # FMTL implementation
│   ├── 📂 simulation/               # SUMO integration
│   │   └── sumo_integration.py      # Traffic simulation utilities
│   └── 📂 utils/                    # Utilities
│       ├── generate_publication_figures.py
│       ├── paper_figures.py
│       └── synthetic_data_generator.py
├── 📂 demo/                         # Interactive demo
│   ├── run_demo.py                  # Main demo orchestrator
│   ├── server.py                    # FL server
│   ├── client.py                    # FL client
│   ├── simple_network.net.xml       # SUMO network
│   ├── routes.rou.xml               # Vehicle routes
│   └── simulation.sumocfg           # SUMO configuration
├── 📂 experiments/                  # Experimental scripts
│   ├── baseline_experiments.py
│   ├── comprehensive_experiments.py
│   └── federated_learning_experiments.py
├── 📂 data/                         # Datasets
│   └── synthetic/                   # Synthetic data for experiments
├── 📂 results/                      # Experimental results
│   ├── experiments/                 # Raw metrics (CSV)
│   └── figures/                     # Visualizations (PDF/PNG)
├── 📂 docs/                         # Documentation
│   ├── technical_reports/           # Technical documentation
│   └── tikz_flowcharts.tex          # LaTeX figures
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🔬 Research Contributions

### 1. Novel FMTL Architecture for IoV
First work to apply federated multi-task learning to dual recommendations (POI + Music) in vehicular networks.

### 2. Context-Aware Client Selection
Multi-objective client selection algorithm considering:
- Spatial proximity
- Temporal patterns
- Mobility similarity
- Data quality

### 3. Privacy-Utility Analysis
Comprehensive evaluation of privacy guarantees (ε-differential privacy) vs. recommendation accuracy.

### 4. IoV-Specific Optimizations
- Handles high mobility and frequent disconnections
- Manages non-IID data distributions
- Reduces communication overhead

### 5. Open-Source Framework
Complete reproducible framework with:
- Synthetic data generation
- SUMO integration
- Visualization tools
- Comprehensive baselines

---

## 📚 Citation

If you use FedRoute in your research, please cite:

```bibtex
@article{fedroute2024,
  title={FedRoute: Privacy-Preserving Federated Multi-Task Learning for Dual Recommendations in Internet of Vehicles},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run linting
flake8 src/

# Run tests (when available)
pytest tests/
```

---

## 📮 Contact

For questions or collaboration opportunities, please open an issue or contact:

- **GitHub**: [@Vatsal-Jha256](https://github.com/Vatsal-Jha256)
- **Repository**: [FedRoute](https://github.com/Vatsal-Jha256/FedRoute)

---

## 🙏 Acknowledgments

- SUMO (Simulation of Urban MObility) for traffic simulation
- PyTorch team for the deep learning framework
- Federated learning research community

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for privacy-preserving IoV research

</div>
