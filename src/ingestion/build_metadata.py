import os
import re
import csv
from pathlib import Path

RAW_DIR = "/scratch/sd5957/finverify_2/data/raw/earnings_calls/cleaned_ECTs_dataset"
OUTPUT_CSV = "/scratch/sd5957/finverify_2/data/processed/metadata.csv"

# Pattern: 2024_Q1_wmt_processed.txt
PATTERN = re.compile(r"(\d{4})_Q(\d)_([a-zA-Z]+)_processed\.txt")

rows = []

for company_dir in Path(RAW_DIR).iterdir():
    if not company_dir.is_dir():
        continue
    company_name = company_dir.name
    
    for txt_file in company_dir.glob("*.txt"):
        match = PATTERN.match(txt_file.name)
        if match:
            year, quarter, ticker = match.groups()
            rows.append({
                "company": company_name,
                "ticker": ticker.upper(),
                "year": int(year),
                "quarter": int(quarter),
                "filename": txt_file.name,
                "filepath": str(txt_file)
            })

# Sort by company, year, quarter
rows.sort(key=lambda x: (x["company"], x["year"], x["quarter"]))

# Write CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["company", "ticker", "year", "quarter", "filename", "filepath"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Total files: {len(rows)}")
print(f"Saved to: {OUTPUT_CSV}")
