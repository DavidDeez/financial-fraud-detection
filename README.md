# Cost-Sensitive Learning for Imbalanced Financial Fraud Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Research project investigating cost-sensitive machine learning techniques for highly imbalanced financial fraud detection (1:1000 class ratio).

## 🎯 Research Overview

This project addresses the critical challenge of class imbalance in financial fraud detection, where traditional classifiers fail to identify the economically significant minority class. We implement and evaluate multiple cost-sensitive approaches to optimize business value rather than pure accuracy.

## 🚀 Key Features

- **Cost-Sensitive Algorithms**: Class-weighted loss functions and sampling techniques
- **Business-Optimal Thresholding**: Custom cost-benefit matrix for decision optimization  
- **Comprehensive Evaluation**: Beyond accuracy - focus on recall, precision, and profit metrics
- **Production-Ready Pipelines**: End-to-end ML workflows with real-time monitoring capabilities

## 📊 Results

| Model | Precision | Recall | F1-Score | Simulated Profit |
|-------|-----------|--------|----------|------------------|
| Baseline (Logistic) | 0.80 | 0.45 | 0.57 | $1,200 |
| Cost-Sensitive RF | **0.65** | **0.87** | **0.74** | **$3,575** |

**Achievements:**
- 86.7% fraud recall while maintaining 65% precision
- 157% profit increase over baseline model
- Effective handling of 1:1000 class imbalance

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/financial-fraud-detection.git
cd financial-fraud-detection
pip install -r requirements.txt
