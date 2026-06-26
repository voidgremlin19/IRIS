# Prompts (prompts/) 

This directory contains all **prompt templates** used by the reasoning models in the Indian Road Intelligence System.

Prompts define how the AI interprets a scene and generates **safe driving decisions**.

---

## Structure

```bash id="q8k2fz"
prompts/
│
└── driving_reason.txt
```

---

## Purpose

The prompts act as the **instruction layer** for the reasoning engine.

They guide the model to:

* Understand traffic context
* Interpret object behavior
* Apply safety rules
* Generate structured driving decisions

---

## driving_reason.txt

This is the core prompt used by the reasoning model.

It defines:

* How the scene is described
* What the model should focus on
* How decisions should be generated

---

## Example Role of Prompt

Input to model:

* Scene graph
* Object detections
* Road conditions

Prompt instructs model to produce:

* Context understanding
* Risk assessment
* Driving action plan

---

## Example Output

```text id="s7g9y1"
Observation:
Truck ahead is braking
Motorcycle is overtaking from right

Decision:
Reduce speed
Increase following distance
Maintain lane
```

---

## Design Guidelines

When editing prompts:

✅ Keep instructions clear and structured
✅ Focus on safety and real-world driving behavior
✅ Ensure outputs are consistent and interpretable

❌ Avoid vague or ambiguous instructions
❌ Do not generate unstructured outputs

---

## Future Improvements

* Multiple prompt templates for different scenarios
* Adaptive prompts based on environment (city, highway, night)
* Safety rule injection for Indian traffic conditions

---

## Summary

Prompts are critical for enabling **context-aware reasoning** and ensuring the system produces **safe and explainable decisions**.

---

**“Prompts define how the AI thinks.”**
