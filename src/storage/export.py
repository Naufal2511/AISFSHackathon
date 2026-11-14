"""Export structured tracking records and cow summaries to CSV, Parquet, and JSON."""

import os
import json
from typing import List, Dict, Any
import pandas as pd
from .database import TrackingDatabase


class TrackingExporter:
    """Exports structured records and cow summaries to disk."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_records(
        self, db: TrackingDatabase, filename_base: str = "tracking_records"
    ) -> Dict[str, str]:
        """Exports records to CSV and Parquet formats."""
        df = db.to_dataframe()
        csv_path = os.path.join(self.output_dir, f"{filename_base}.csv")
        df.to_csv(csv_path, index=False)

        parquet_path = os.path.join(self.output_dir, f"{filename_base}.parquet")
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception:
            # Fallback if pyarrow is not installed
            parquet_path = ""

        return {"csv": csv_path, "parquet": parquet_path}

    def export_cow_summary(
        self, summary: List[Dict[str, Any]], filename: str = "cow_identity_summary.json"
    ) -> str:
        """Exports individual cow identity metrics to formatted JSON and CSV."""
        json_path = os.path.join(self.output_dir, filename)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        csv_path = os.path.join(self.output_dir, filename.replace(".json", ".csv"))
        df = pd.DataFrame(summary)
        df.to_csv(csv_path, index=False)

        return json_path
