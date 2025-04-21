# === log_processor.py ===
import re
from langchain.schema import Document
import uuid
# Finalized regex pattern
log_regex = re.compile(r"""
    ^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})          # Timestamp
    \s+
    (?:\[(?P<level1>[A-Z]+)\]|(?P<level2>[A-Z]+))                 # [LEVEL] or LEVEL
    (?:\s+\[(?P<component>[^\]]+)\])?                             # Optional [Component]
    \s+(?P<message>[^\n]+)                                        # First line of message
    (?P<exception>(?:\n\s+(?!\d{4}-\d{2}-\d{2}).*)*)               # Multi-line stack trace
    """, re.VERBOSE | re.MULTILINE)


def parse_log_entry(entry: str):
    match = log_regex.match(entry.strip())
    print(f"Processing entry: {entry.strip()}");
    print(f"match: {match}");
    if not match:
        return None

    timestamp = match.group("timestamp")
    level = match.group("level1") or match.group("level2")
    component = match.group("component") or "Unknown"
    message = match.group("message").strip()
    exception = match.group("exception").strip() if match.group("exception") else ""

    full_content = f"{message}\n{exception}" if exception else message

    return Document(
        page_content=full_content,
        metadata={
            "id": str(uuid.uuid4()),
            "timestamp": timestamp,
            "level": level,
            "component": component,
            "message": message,
            "exception": exception,
        }
    )


def split_log_entries(raw_log_data: str):
    return re.split(r"(?=^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", raw_log_data, flags=re.MULTILINE)


def process_logs(file_path: str):
    with open(file_path, "r") as file:
        raw_data = file.read()

    entries = split_log_entries(raw_data)
    documents = [parse_log_entry(entry) for entry in entries]
    return [doc for doc in documents if doc is not None]
