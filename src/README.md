# Core Engine (src/) 

This directory contains the **core intelligence layer** of the Indian Road Intelligence System.

It implements the complete **AI Driver Brain pipeline**, transforming raw visual input into structured reasoning and actionable decisions.

---

## Structure

```bash
src/
│
├── __init__.py
├── detect.py
├── graph.py
├── reason.py
├── pipeline.py
├── viz.py
└── config.py
```

---

## Module Overview

### detect.py — Perception Layer

Handles object detection using models like YOLO or RT-DETR.

Responsibilities:

* Detect vehicles, pedestrians, and road elements
* Estimate positions and distances
* Output structured detections

Answers: **What is there?**

---

### graph.py — Scene Graph Builder

Converts raw detections into a structured scene representation.

Responsibilities:

* Define spatial relationships (left, right, ahead)
* Organize objects into a scene graph
* Represent environment context

Answers: **How are objects related?**

---

### reason.py — Reasoning Engine

Integrates Vision-Language and Language Models.

Responsibilities:

* Interpret scene context
* Analyze behavior and intent
* Generate high-level decisions

Answers: **What is happening and what should be done?**

---

### pipeline.py — Orchestrator

Coordinates the full processing pipeline.

Flow:

Detection → Scene Graph → Reasoning → Decision

Responsibilities:

* Connect all modules
* Manage data flow
* Produce final output

---

### viz.py — Visualization

Provides utilities for visual debugging.

Responsibilities:

* Draw bounding boxes
* Overlay scene graphs
* Display reasoning outputs

---

### config.py — Configuration

Centralized configuration for the system.

Includes:

* Model settings
* API keys (via environment variables)
* Prompt templates
* Threshold values

---

## End-to-End Flow

```text
Input Image
    ↓
detect.py
    ↓
graph.py
    ↓
reason.py
    ↓
pipeline.py
    ↓
Decision Output
```

---

## Design Principles

* **Modular** → Each component is independent
* **Composable** → Easily extend or replace modules
* **Explainable** → Every step is interpretable
* **Scalable** → Ready for real-time deployment

---

## Usage

Typical usage via pipeline:

```python
from src.pipeline import Pipeline

pipeline = Pipeline()
result = pipeline.run(image)

print(result)
```

---

## Purpose

This layer acts as the **central intelligence system**, enabling:

* Scene understanding
* Context-aware reasoning
* Safe decision generation

---

## Summary

The `src/` directory is the **brain of the system**, where perception evolves into reasoning.

---

**“From pixels to decisions — this is where intelligence happens.”**


