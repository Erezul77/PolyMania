#!/usr/bin/env python3
"""Create Grafana TimescaleDB datasource."""
import os
import requests
import json

ds = {
    "name": "TimescaleDB",
    "type": "grafana-postgresql-datasource",
    "access": "proxy",
    "url": "timescaledb:5432",
    "user": os.getenv("DB_USER", "polymania"),
    "database": os.getenv("DB_NAME", "polymania"),
    "isDefault": True,
    "secureJsonData": {"password": os.getenv("DB_PASSWORD", "")},
    "jsonData": {
        "sslmode": "disable",
        "postgresVersion": 1500,
        "timescaledb": True
    }
}

r = requests.post(
    "http://localhost:3000/api/datasources",
    auth=(os.getenv("GRAFANA_USER", "admin"), os.getenv("GRAFANA_PASSWORD", "")),
    json=ds
)
print(f"Status: {r.status_code}")
print(f"Response: {r.text}")
