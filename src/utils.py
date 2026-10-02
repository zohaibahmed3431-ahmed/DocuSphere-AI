from pathlib import Path


def extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_code_file(filename: str) -> bool:
    return extension(filename) in {
        ".py", ".java", ".cpp", ".c", ".h", ".hpp", ".js", ".ts",
        ".html", ".css", ".scss", ".sql", ".json", ".xml", ".yaml", ".yml", ".toml",
        ".ini", ".cfg", ".conf", ".sh", ".bat", ".ps1", ".php", ".go", ".rs", ".rb",
        ".kt", ".kts", ".swift", ".r", ".m", ".vue", ".svelte", ".tex",
    }
