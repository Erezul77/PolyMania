#!/usr/bin/env python3
"""Test database connection."""
import asyncio
import asyncpg
import os

async def test():
    try:
        conn = await asyncpg.connect(
            host=os.getenv("POSTGRES_HOST", "timescaledb"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "polymania"),
            password=os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "")),
            database=os.getenv("POSTGRES_DB", "polymania")
        )
        result = await conn.fetchval("SELECT 1")
        print(f"Database connection OK: {result}")
        await conn.close()
    except Exception as e:
        print(f"Database connection FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test())
