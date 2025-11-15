import os
from pathlib import Path
from typing import Dict, List

class DataScanner:
    def __init__(self, base_path: str = "data/legacy_source"):
        self.base_path = Path(base_path)
        self.file_types = {
            'gex_json': '**/*.json',
            'tick_csv': '**/*.csv',
            'depth_jsonl': '**/*.jsonl',
            'sqlite_db': '**/*.db'
        }

    def scan_directory(self) -> Dict[str, List[Path]]:
        """Scan the base directory and categorize files by type."""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path {self.base_path} does not exist")

        categorized_files = {
            'gex_json': [],
            'tick_csv': [],
            'depth_jsonl': [],
            'sqlite_db': []
        }

        for file_type, pattern in self.file_types.items():
            files = list(self.base_path.glob(pattern))
            categorized_files[file_type].extend(files)

        return categorized_files

    def get_file_stats(self, files: Dict[str, List[Path]]) -> Dict[str, int]:
        """Get statistics about discovered files."""
        return {file_type: len(file_list) for file_type, file_list in files.items()}

    def validate_files(self, files: Dict[str, List[Path]]) -> List[str]:
        """Validate that files exist and are readable."""
        errors = []
        for file_type, file_list in files.items():
            for file_path in file_list:
                if not file_path.exists():
                    errors.append(f"{file_type} file {file_path} does not exist")
                elif not os.access(file_path, os.R_OK):
                    errors.append(f"{file_type} file {file_path} is not readable")
        return errors

    def get_sample_files(self, files: Dict[str, List[Path]], sample_size: int = 5) -> Dict[str, List[Path]]:
        """Get sample files for each type for testing."""
        samples = {}
        for file_type, file_list in files.items():
            samples[file_type] = file_list[:sample_size]
        return samples
