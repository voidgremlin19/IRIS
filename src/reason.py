"""
reason.py — Reasoning Engine
AI-powered scene reasoning with rule-based fallback.
Generates driving decisions from scene context.
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

from src.config import (
    USE_LLM_REASONING,
    LLM_API_KEY,
    LLM_MODEL,
    PROMPTS_DIR,
    RISK_LEVELS,
    HEAVY_VEHICLES,
    VULNERABLE_ROAD_USERS,
)

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Reasoning engine that interprets scene data and generates driving decisions.
    Supports:
      - Rule-based reasoning (default, no external API needed)
      - LLM-based reasoning (optional, requires API key)
    """

    def __init__(self, use_llm: bool = USE_LLM_REASONING):
        self.use_llm = use_llm and bool(LLM_API_KEY)
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load the driving reasoning prompt template."""
        prompt_path = PROMPTS_DIR / "driving_reason.txt"
        try:
            return prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("Prompt file not found, using default template.")
            return self._default_prompt()

    def _default_prompt(self) -> str:
        return """You are an AI driving assistant for Indian roads.
Given the scene information, analyze the situation and provide:
1. Context: What is happening in the scene
2. Decision: What the driver should do
3. Risk Level: none/low/medium/high/critical
4. Voice Alert: A short spoken warning

Scene Data:
{scene_data}

Respond in JSON format with keys: context, decision, risk, alert"""

    def reason(
        self,
        detections: dict,
        scene_graph: dict,
        distances: dict,
    ) -> dict:
        """
        Generate driving decision from scene analysis.
        
        Args:
            detections: Object detection results
            scene_graph: Scene graph with spatial relationships
            distances: Distance estimation results
            
        Returns:
            Reasoning output with context, decision, risk, and alert
        """
        if self.use_llm:
            return self._llm_reason(detections, scene_graph, distances)
        else:
            return self._rule_based_reason(detections, scene_graph, distances)

    def _rule_based_reason(
        self,
        detections: dict,
        scene_graph: dict,
        distances: dict,
    ) -> dict:
        """
        Rule-based reasoning engine optimized for Indian traffic.
        No external API needed — works fully offline.
        """
        objects = detections.get("objects", [])
        ego_view = scene_graph.get("ego_view", {})
        dist_summary = distances.get("summary", {})
        distance_list = distances.get("distances", [])
        clusters = scene_graph.get("clusters", [])
        relations = scene_graph.get("relations", [])

        # ─── Analyze Scene Context ───────────────────────────────
        context_parts = []
        decisions = []
        risk_score = 0
        alerts = []

        # Count object types
        type_counts = {}
        for obj in objects:
            t = obj["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        # ─── Rule 1: Heavy Vehicles Ahead ────────────────────────
        ahead_objects = ego_view.get("ahead", [])
        ahead_heavy = [o for o in ahead_objects if o["type"] in HEAVY_VEHICLES]
        if ahead_heavy:
            context_parts.append(
                f"{ahead_heavy[0]['type']} detected ahead"
            )
            decisions.append("slow_down")
            risk_score += 3
            alerts.append(f"{ahead_heavy[0]['type'].replace('_', ' ').title()} ahead. Slow down.")

        # ─── Rule 2: Vulnerable Road Users ────────────────────────
        vru_near = []
        for d in distance_list:
            if d["type"] in VULNERABLE_ROAD_USERS and d["zone"] in ("near", "medium"):
                vru_near.append(d)

        if vru_near:
            vru_types = list(set(v["type"] for v in vru_near))
            context_parts.append(
                f"Vulnerable road users nearby: {', '.join(vru_types)}"
            )
            decisions.append("extreme_caution")
            risk_score += 4
            
            closest_vru = min(vru_near, key=lambda v: v["estimated_meters"])
            alerts.append(
                f"Caution! {closest_vru['type'].replace('_', ' ').title()} "
                f"at approximately {closest_vru['estimated_meters']:.0f} meters."
            )

        # ─── Rule 3: Animals on Road ─────────────────────────────
        animal_types = {"cow", "stray_dog", "horse", "elephant", "sheep", "cat"}
        animals_detected = [o for o in objects if o["type"] in animal_types]
        if animals_detected:
            animal_names = list(set(a["type"] for a in animals_detected))
            context_parts.append(
                f"Animals on road: {', '.join(animal_names)}"
            )
            decisions.append("slow_down")
            risk_score += 3
            alerts.append(
                f"Warning! {animal_names[0].replace('_', ' ').title()} on the road. "
                f"Reduce speed immediately."
            )

        # ─── Rule 4: Congestion Detection ────────────────────────
        if clusters:
            max_cluster = max(clusters, key=lambda c: c["size"])
            if max_cluster["size"] >= 3:
                context_parts.append(
                    f"Traffic congestion detected with {max_cluster['size']} objects clustered"
                )
                decisions.append("slow_down")
                risk_score += 2
                alerts.append("Traffic congestion ahead. Reduce speed.")

        # ─── Rule 5: Overtaking Detection ────────────────────────
        right_objects = ego_view.get("right_side", [])
        fast_right = [o for o in right_objects if o["type"] in ("motorcycle/bike", "car", "bicycle")]
        if fast_right:
            context_parts.append(
                f"{fast_right[0]['type']} on the right side, possible overtaking"
            )
            decisions.append("maintain_lane")
            risk_score += 1
            alerts.append(f"Vehicle overtaking from right. Maintain lane.")

        # ─── Rule 6: Multiple Objects Close ──────────────────────
        near_count = dist_summary.get("near_objects", 0)
        if near_count >= 3:
            context_parts.append(f"{near_count} objects in close proximity")
            decisions.append("emergency_caution")
            risk_score += 4
            alerts.append("Multiple objects very close. Drive with extreme caution.")

        # ─── Rule 7: Clear Road ──────────────────────────────────
        if not objects or (len(objects) <= 1 and dist_summary.get("near_objects", 0) == 0):
            context_parts.append("Road appears relatively clear")
            decisions.append("proceed_normally")
            alerts.append("Road is clear. Proceed safely.")

        # ─── Aggregate Decision ──────────────────────────────────
        # Priority: emergency_brake > extreme_caution > slow_down > caution > proceed
        decision_priority = {
            "emergency_brake": 5,
            "emergency_caution": 4,
            "extreme_caution": 4,
            "slow_down": 3,
            "maintain_lane": 2,
            "caution": 2,
            "proceed_normally": 0,
        }

        if decisions:
            final_decision = max(decisions, key=lambda d: decision_priority.get(d, 0))
        else:
            final_decision = "proceed_normally"

        # Determine risk level
        if risk_score >= 8:
            risk = "critical"
        elif risk_score >= 5:
            risk = "high"
        elif risk_score >= 3:
            risk = "medium"
        elif risk_score >= 1:
            risk = "low"
        else:
            risk = "none"

        # Build context string
        context = ". ".join(context_parts) if context_parts else "Normal driving conditions"

        # Select most urgent alert
        primary_alert = alerts[0] if alerts else "All clear. Drive safely."

        return {
            "context": context,
            "decision": final_decision,
            "risk": risk,
            "risk_score": risk_score,
            "alert": primary_alert,
            "all_alerts": alerts,
            "reasoning_method": "rule_based",
            "object_summary": type_counts,
            "scene_complexity": self._assess_complexity(objects, clusters),
        }

    def _assess_complexity(self, objects: list, clusters: list) -> str:
        """Assess overall scene complexity."""
        obj_count = len(objects)
        cluster_count = len(clusters)

        if obj_count >= 8 or cluster_count >= 3:
            return "very_high"
        elif obj_count >= 5 or cluster_count >= 2:
            return "high"
        elif obj_count >= 3:
            return "moderate"
        elif obj_count >= 1:
            return "low"
        else:
            return "empty"

    def _llm_reason(
        self,
        detections: dict,
        scene_graph: dict,
        distances: dict,
    ) -> dict:
        """
        LLM-based reasoning using OpenAI-compatible API.
        Falls back to rule-based if API call fails.
        """
        try:
            import openai

            scene_data = json.dumps({
                "detections": detections.get("objects", []),
                "scene_graph": {
                    "relations": scene_graph.get("relations", [])[:10],
                    "ego_view": scene_graph.get("ego_view", {}),
                    "clusters": scene_graph.get("clusters", []),
                },
                "distances": {
                    "summary": distances.get("summary", {}),
                    "details": distances.get("distances", []),
                },
            }, indent=2)

            prompt = self.prompt_template.replace("{scene_data}", scene_data)

            client = openai.OpenAI(api_key=LLM_API_KEY)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI driving assistant "
                                   "specialized in Indian road conditions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            result["reasoning_method"] = "llm"
            return result

        except Exception as e:
            logger.warning(f"LLM reasoning failed: {e}. Falling back to rule-based.")
            return self._rule_based_reason(detections, scene_graph, distances)


def generate_reasoning(
    detections: dict,
    scene_graph: dict,
    distances: dict,
) -> dict:
    """Convenience function for reasoning."""
    engine = ReasoningEngine()
    return engine.reason(detections, scene_graph, distances)
