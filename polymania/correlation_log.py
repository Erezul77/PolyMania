import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


CLUSTERS_CSV_PATH = "correlation_clusters.csv"


def _ensure_header(path: str) -> None:
    """
    Ensure the CSV file exists and has a header row.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    header = [
        "timestamp_iso",
        "topic",
        "cluster_direction",
        "count",
        "up_count",
        "down_count",
        "avg_abs_jump",
        "avg_dominance",
        "window_sec",
        "event_ids",
        "event_titles",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_correlation_row(ts: int, cluster: Dict[str, Any]) -> None:
    """
    Append a single correlation-cluster row to correlation_clusters.csv
    for later offline analysis.
    """
    _ensure_header(CLUSTERS_CSV_PATH)

    timestamp_iso = datetime.utcfromtimestamp(ts).isoformat() + "Z"

    events: List[Dict[str, Any]] = cluster.get("events") or []
    ids = " | ".join(str(e.get("event_id") or "") for e in events)
    titles = " | ".join(str(e.get("event_title") or "") for e in events)

    row = [
        timestamp_iso,
        cluster.get("topic"),
        cluster.get("cluster_direction"),
        cluster.get("count"),
        cluster.get("up_count"),
        cluster.get("down_count"),
        f"{float(cluster.get('avg_abs_jump') or 0.0):.6f}",
        f"{float(cluster.get('avg_dominance') or 0.0):.6f}",
        cluster.get("window_sec"),
        ids,
        titles,
    ]

    with open(CLUSTERS_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
