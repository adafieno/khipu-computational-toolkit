"""
Khipu metadata loader.

Extracts top-level khipu information from the khipu_main and primary_cord tables.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


class KhipuLoader:
    """Load and query khipu metadata from the OKR database."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the loader.
        
        Args:
            db_path: Path to khipu.db. If None, looks in project root.
        """
        if db_path is None:
            # Default to project root
            db_path = Path(__file__).parent.parent.parent / "khipu.db"
        
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
    
    def _connect(self) -> sqlite3.Connection:
        """Create database connection."""
        return sqlite3.connect(self.db_path)
    
    def get_all_khipus(self) -> pd.DataFrame:
        """
        Load all khipu metadata.
        
        Returns:
            DataFrame with khipu_main columns
        """
        with self._connect() as conn:
            query = "SELECT * FROM khipu_main ORDER BY KHIPU_ID"
            return pd.read_sql_query(query, conn)
    
    def get_khipu_by_id(self, khipu_id: int) -> Optional[Dict[str, Any]]:
        """
        Load a single khipu by ID.
        
        Args:
            khipu_id: The KHIPU_ID to retrieve
            
        Returns:
            Dictionary of khipu metadata or None if not found
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM khipu_main WHERE KHIPU_ID = ?",
                (khipu_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
    
    def get_khipu_by_okr_num(self, okr_num: str) -> Optional[Dict[str, Any]]:
        """
        Load a khipu by its OKR number (e.g., "KH0255").
        
        Args:
            okr_num: The OKR_NUM identifier
            
        Returns:
            Dictionary of khipu metadata or None if not found
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM khipu_main WHERE OKR_NUM = ?",
                (okr_num,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
    
    def get_khipus_by_provenance(self, provenance: str) -> pd.DataFrame:
        """
        Get all khipus from a specific provenance.
        
        Args:
            provenance: The provenance name
            
        Returns:
            DataFrame of matching khipus
        """
        with self._connect() as conn:
            query = """
                SELECT * FROM khipu_main 
                WHERE PROVENANCE = ? 
                ORDER BY KHIPU_ID
            """
            return pd.read_sql_query(query, conn, params=(provenance,))
    
    def get_khipus_by_region(self, region: str) -> pd.DataFrame:
        """
        Get all khipus from a specific region.
        
        Args:
            region: The region name
            
        Returns:
            DataFrame of matching khipus
        """
        with self._connect() as conn:
            query = """
                SELECT * FROM khipu_main 
                WHERE REGION = ? 
                ORDER BY KHIPU_ID
            """
            return pd.read_sql_query(query, conn, params=(region,))
    
    def get_primary_cord(self, khipu_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the primary cord for a khipu.
        
        Args:
            khipu_id: The KHIPU_ID
            
        Returns:
            Dictionary of primary cord data or None if not found
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM primary_cord WHERE KHIPU_ID = ?",
                (khipu_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about the database.
        
        Returns:
            Dictionary with counts and statistics
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count khipus
            cursor.execute("SELECT COUNT(*) FROM khipu_main")
            stats['total_khipus'] = cursor.fetchone()[0]
            
            # Count khipus with provenance
            cursor.execute(
                "SELECT COUNT(*) FROM khipu_main WHERE PROVENANCE IS NOT NULL"
            )
            stats['khipus_with_provenance'] = cursor.fetchone()[0]
            
            # Count cords
            cursor.execute("SELECT COUNT(*) FROM cord")
            stats['total_cords'] = cursor.fetchone()[0]
            
            # Count knots
            cursor.execute("SELECT COUNT(*) FROM knot")
            stats['total_knots'] = cursor.fetchone()[0]
            
            # Count regions
            cursor.execute(
                "SELECT COUNT(DISTINCT REGION) FROM khipu_main WHERE REGION IS NOT NULL"
            )
            stats['unique_regions'] = cursor.fetchone()[0]
            
            # Count provenances
            cursor.execute(
                "SELECT COUNT(DISTINCT PROVENANCE) FROM khipu_main WHERE PROVENANCE IS NOT NULL"
            )
            stats['unique_provenances'] = cursor.fetchone()[0]
            
            return stats
    
    def search_khipus(
        self,
        provenance: Optional[str] = None,
        region: Optional[str] = None,
        museum: Optional[str] = None,
        complete_only: bool = False
    ) -> pd.DataFrame:
        """
        Search khipus with multiple filters.
        
        Args:
            provenance: Filter by provenance (exact match)
            region: Filter by region (exact match)
            museum: Filter by museum name (partial match)
            complete_only: Only return complete khipus
            
        Returns:
            DataFrame of matching khipus
        """
        with self._connect() as conn:
            conditions = []
            params = []
            
            if provenance:
                conditions.append("PROVENANCE = ?")
                params.append(provenance)
            
            if region:
                conditions.append("REGION = ?")
                params.append(region)
            
            if museum:
                conditions.append("MUSEUM_NAME LIKE ?")
                params.append(f"%{museum}%")
            
            if complete_only:
                conditions.append("COMPLETE = 1.0")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT * FROM khipu_main 
                WHERE {where_clause}
                ORDER BY KHIPU_ID
            """
            
            return pd.read_sql_query(query, conn, params=params)


if __name__ == "__main__":
    # Example usage
    loader = KhipuLoader()
    
    print("Database Summary:")
    stats = loader.get_summary_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    print("\nFirst 5 khipus:")
    khipus = loader.get_all_khipus()
    print(khipus.head()[['KHIPU_ID', 'OKR_NUM', 'PROVENANCE', 'REGION']])
    
    print("\nExample khipu (ID=1000000):")
    khipu = loader.get_khipu_by_id(1000000)
    if khipu:
        print(f"  OKR_NUM: {khipu['OKR_NUM']}")
        print(f"  Provenance: {khipu['PROVENANCE']}")
        print(f"  Region: {khipu['REGION']}")
