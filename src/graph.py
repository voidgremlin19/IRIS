"""
graph.py — Scene Graph Builder
Converts raw detections into spatial relationships for scene understanding.
Answers: How are objects related to each other?
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SceneGraphBuilder:
    """
    Builds a scene graph from object detections.
    Determines spatial relationships (left_of, right_of, ahead, behind, near)
    between detected objects.
    """

    # Horizontal position thresholds (normalized 0-1)
    LEFT_ZONE = 0.35
    RIGHT_ZONE = 0.65

    # Vertical position thresholds (normalized 0-1, lower y = higher in image = further away)
    AHEAD_ZONE = 0.45
    BEHIND_ZONE = 0.70

    # Proximity threshold for "near" relationship
    PROXIMITY_THRESHOLD = 0.15

    def build(self, detections: dict) -> dict:
        """
        Build scene graph from detections.
        
        Args:
            detections: Output from detect.py with 'objects' list
            
        Returns:
            Scene graph with relations, zones, and ego perspective
        """
        objects = detections.get("objects", [])

        if not objects:
            return {
                "relations": [],
                "zones": {"left": [], "center": [], "right": []},
                "ego_view": {
                    "ahead": [],
                    "left_side": [],
                    "right_side": [],
                    "close": [],
                },
                "object_count": 0,
            }

        # Assign unique names to each object
        type_counts = {}
        for obj in objects:
            t = obj["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
            obj["name"] = f"{t}_{type_counts[t]}"

        # Classify objects into zones
        zones = self._classify_zones(objects)

        # Build pairwise spatial relations
        relations = self._build_relations(objects)

        # Build ego-centric view (from driver's perspective)
        ego_view = self._build_ego_view(objects)

        # Identify clusters (groups of nearby objects)
        clusters = self._identify_clusters(objects)

        return {
            "relations": relations,
            "zones": zones,
            "ego_view": ego_view,
            "clusters": clusters,
            "object_count": len(objects),
        }

    def _classify_zones(self, objects: List[Dict]) -> dict:
        """Classify objects into left/center/right zones."""
        zones = {"left": [], "center": [], "right": []}

        for obj in objects:
            norm = obj.get("bbox_normalized", {})
            cx = norm.get("cx", 0.5)

            obj_summary = {
                "type": obj["type"],
                "name": obj.get("name", obj["type"]),
                "confidence": obj["confidence"],
                "position": {"cx": cx, "cy": norm.get("cy", 0.5)},
            }

            if cx < self.LEFT_ZONE:
                zones["left"].append(obj_summary)
            elif cx > self.RIGHT_ZONE:
                zones["right"].append(obj_summary)
            else:
                zones["center"].append(obj_summary)

        return zones

    def _build_relations(self, objects: List[Dict]) -> List[Dict]:
        """Build pairwise spatial relationships between objects."""
        relations = []

        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i >= j:
                    continue

                norm_a = obj_a.get("bbox_normalized", {})
                norm_b = obj_b.get("bbox_normalized", {})

                cx_a, cy_a = norm_a.get("cx", 0.5), norm_a.get("cy", 0.5)
                cx_b, cy_b = norm_b.get("cx", 0.5), norm_b.get("cy", 0.5)

                dx = cx_b - cx_a
                dy = cy_b - cy_a

                # Determine primary spatial relation
                if abs(dx) > abs(dy):
                    # Horizontal relationship dominates
                    if dx > 0.05:
                        relation = "left_of"
                    elif dx < -0.05:
                        relation = "right_of"
                    else:
                        relation = "aligned_with"
                else:
                    # Vertical relationship dominates
                    if dy > 0.05:
                        relation = "ahead_of"  # lower in image = closer
                    elif dy < -0.05:
                        relation = "behind"
                    else:
                        relation = "aligned_with"

                relations.append({
                    "subject": obj_a.get("name", obj_a["type"]),
                    "relation": relation,
                    "object": obj_b.get("name", obj_b["type"]),
                    "distance": round(((dx ** 2) + (dy ** 2)) ** 0.5, 3),
                })

                # Check for proximity
                dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
                if dist < self.PROXIMITY_THRESHOLD:
                    relations.append({
                        "subject": obj_a.get("name", obj_a["type"]),
                        "relation": "near",
                        "object": obj_b.get("name", obj_b["type"]),
                        "distance": round(dist, 3),
                    })

        return relations

    def _build_ego_view(self, objects: List[Dict]) -> dict:
        """Build ego-centric view from the driver's perspective."""
        ego = {
            "ahead": [],
            "left_side": [],
            "right_side": [],
            "close": [],
            "far": [],
        }

        for obj in objects:
            norm = obj.get("bbox_normalized", {})
            cx = norm.get("cx", 0.5)
            cy = norm.get("cy", 0.5)
            area = obj.get("area_ratio", 0)

            obj_info = {
                "type": obj["type"],
                "name": obj.get("name", obj["type"]),
                "confidence": obj["confidence"],
                "size_indicator": "large" if area > 0.05 else "medium" if area > 0.02 else "small",
            }

            # Ego-centric zone classification
            if self.LEFT_ZONE <= cx <= self.RIGHT_ZONE:
                ego["ahead"].append(obj_info)
            if cx < self.LEFT_ZONE:
                ego["left_side"].append(obj_info)
            if cx > self.RIGHT_ZONE:
                ego["right_side"].append(obj_info)

            # Distance based on position and size
            if cy > self.BEHIND_ZONE or area > 0.08:
                ego["close"].append(obj_info)
            elif cy < self.AHEAD_ZONE and area < 0.02:
                ego["far"].append(obj_info)

        return ego

    def _identify_clusters(self, objects: List[Dict]) -> List[Dict]:
        """Identify clusters of nearby objects (potential congestion/crowd zones)."""
        clusters = []
        used = set()

        for i, obj_a in enumerate(objects):
            if i in used:
                continue

            cluster_members = [obj_a.get("name", obj_a["type"])]
            used.add(i)

            norm_a = obj_a.get("bbox_normalized", {})
            cx_a, cy_a = norm_a.get("cx", 0.5), norm_a.get("cy", 0.5)

            for j, obj_b in enumerate(objects):
                if j in used:
                    continue

                norm_b = obj_b.get("bbox_normalized", {})
                cx_b, cy_b = norm_b.get("cx", 0.5), norm_b.get("cy", 0.5)

                dist = ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5
                if dist < self.PROXIMITY_THRESHOLD * 2:
                    cluster_members.append(obj_b.get("name", obj_b["type"]))
                    used.add(j)

            if len(cluster_members) > 1:
                clusters.append({
                    "members": cluster_members,
                    "size": len(cluster_members),
                    "type": "congestion" if len(cluster_members) > 3 else "group",
                })

        return clusters


def build_scene_graph(detections: dict) -> dict:
    """Convenience function for scene graph construction."""
    builder = SceneGraphBuilder()
    return builder.build(detections)
