"""
Script to analyze geographical and provenance metadata in the OKR database.
"""

import sys
from pathlib import Path

# Add src directory to path for config import
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import get_config  # noqa: E402 # type: ignore

import sqlite3  # noqa: E402


def analyze_geography(db_path):
    """Analyze geographical distribution of khipus."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 80)
    print("GEOGRAPHICAL METADATA ANALYSIS")
    print("=" * 80)
    print()

    # Provenance distribution
    print("TOP PROVENANCES (by khipu count):")
    print("-" * 80)
    cursor.execute("""
        SELECT PROVENANCE, REGION, COUNT(*) as count
        FROM khipu_main
        WHERE PROVENANCE IS NOT NULL
        GROUP BY PROVENANCE, REGION
        ORDER BY count DESC
        LIMIT 20;
    """)

    for prov, region, count in cursor.fetchall():
        region_str = region if region else "Unknown"
        print(f"  {prov:<45} | {region_str:<20} | {count:>3} khipus")

    print()
    print("=" * 80)

    # Region summary
    print("\nREGION SUMMARY:")
    print("-" * 80)
    cursor.execute("""
        SELECT REGION, COUNT(*) as count
        FROM khipu_main
        WHERE REGION IS NOT NULL
        GROUP BY REGION
        ORDER BY count DESC;
    """)

    for region, count in cursor.fetchall():
        print(f"  {region:<30} | {count:>3} khipus")

    print()
    print("=" * 80)

    # Check regions_dc table for more detail
    print("\nREGIONS DICTIONARY (provenance → region mapping):")
    print("-" * 80)
    cursor.execute("""
        SELECT provenance, region, north_south
        FROM regions_dc
        ORDER BY north_south, region;
    """)

    print(f"{'Provenance':<45} | {'Region':<20} | N/S")
    print("-" * 80)
    for prov, region, ns in cursor.fetchall():
        print(f"{prov:<45} | {region:<20} | {ns}")

    print()
    print("=" * 80)

    # Museum locations
    print("\nMUSEUM LOCATIONS (top 15):")
    print("-" * 80)
    cursor.execute("""
        SELECT MUSEUM_NAME, COUNT(*) as count
        FROM khipu_main
        WHERE MUSEUM_NAME IS NOT NULL
        GROUP BY MUSEUM_NAME
        ORDER BY count DESC
        LIMIT 15;
    """)

    for museum, count in cursor.fetchall():
        print(f"  {museum:<60} | {count:>3} khipus")

    print()
    print("=" * 80)

    # Discovered by
    print("\nRESEARCHERS/DISCOVERERS:")
    print("-" * 80)
    cursor.execute("""
        SELECT DISCOVERED_BY, COUNT(*) as count
        FROM khipu_main
        WHERE DISCOVERED_BY IS NOT NULL AND DISCOVERED_BY != ''
        GROUP BY DISCOVERED_BY
        ORDER BY count DESC
        LIMIT 10;
    """)

    rows = cursor.fetchall()
    if rows:
        for discoverer, count in rows:
            print(f"  {discoverer:<40} | {count:>3} khipus")
    else:
        print("  (No discoverer data available)")

    print()
    print("=" * 80)

    # Sample detailed record
    print("\nSAMPLE DETAILED RECORD:")
    print("-" * 80)
    cursor.execute("""
        SELECT KHIPU_ID, OKR_NUM, PROVENANCE, REGION,
               MUSEUM_NAME, MUSEUM_NUM, DISCOVERED_BY, DATE_DISCOVERED
        FROM khipu_main
        WHERE PROVENANCE IS NOT NULL
        LIMIT 1;
    """)

    row = cursor.fetchone()
    if row:
        khipu_id, okr_num, prov, region, museum, museum_num, disc_by, disc_date = row
        print(f"  KHIPU_ID: {khipu_id}")
        print(f"  OKR_NUM: {okr_num}")
        print(f"  PROVENANCE: {prov}")
        print(f"  REGION: {region}")
        print(f"  MUSEUM: {museum}")
        print(f"  MUSEUM_NUM: {museum_num}")
        print(f"  DISCOVERED_BY: {disc_by}")
        print(f"  DATE_DISCOVERED: {disc_date}")

    conn.close()


if __name__ == "__main__":
    config = get_config()
    analyze_geography(config.get_database_path())
