from typing import Dict, Any, List
from datetime import datetime
import uuid
import json


class LineageTracker:
    def __init__(self, lineage_file: str = "data_lineage.json"):
        self.lineage_file = lineage_file
        self.lineage_records = self.load_lineage()

    def load_lineage(self) -> List[Dict[str, Any]]:
        """Load existing lineage records from file."""
        try:
            with open(self.lineage_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_lineage(self):
        """Save lineage records to file."""
        with open(self.lineage_file, "w") as f:
            json.dump(self.lineage_records, f, indent=2, default=str)

    def record_import(
        self, source_file: str, record_count: int, import_type: str
    ) -> str:
        """Record a data import event."""
        import_id = str(uuid.uuid4())
        record = {
            "import_id": import_id,
            "source_file": source_file,
            "import_timestamp": datetime.now().isoformat(),
            "record_count": record_count,
            "import_type": import_type,
            "status": "completed",
        }
        self.lineage_records.append(record)
        self.save_lineage()
        return import_id

    def get_import_history(self, source_file: str = None) -> List[Dict[str, Any]]:
        """Get import history, optionally filtered by source file."""
        if source_file:
            return [r for r in self.lineage_records if r["source_file"] == source_file]
        return self.lineage_records

    def get_stats(self) -> Dict[str, Any]:
        """Get lineage statistics."""
        total_imports = len(self.lineage_records)
        total_records = sum(r.get("record_count", 0) for r in self.lineage_records)
        import_types = {}
        for r in self.lineage_records:
            itype = r.get("import_type", "unknown")
            import_types[itype] = import_types.get(itype, 0) + 1

        return {
            "total_imports": total_imports,
            "total_records": total_records,
            "import_types": import_types,
        }
