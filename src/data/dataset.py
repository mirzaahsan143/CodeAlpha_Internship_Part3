"""
Dataset investigation and analysis module.
Handles dataset discovery, integrity checks, and statistics generation.
"""
import os
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    DATA_DIR, DATA_DIR_ALT, CLASS_NAMES, NUM_CLASSES,
    HEALTHY_KEYWORDS, IMG_SIZE
)


class LeafDataset:
    """
    Comprehensive dataset handler for the PlantVillage leaf disease dataset.
    Handles auto-discovery, integrity checks, statistics, and loading.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        if not self.data_dir.exists():
            # Fall back to alternate directory
            self.data_dir = DATA_DIR_ALT

        self.class_names: List[str] = []
        self.num_classes: int = 0
        self.file_paths: Dict[str, List[Path]] = {}
        self.class_counts: Dict[str, int] = {}
        self.dataset_summary: Dict = {}

        # Discover the dataset
        self._discover_classes()

    def _discover_classes(self) -> None:
        """Auto-discover all class directories in the dataset folder."""
        items = sorted(os.listdir(self.data_dir))
        self.class_names = [
            d for d in items
            if os.path.isdir(self.data_dir / d)
        ]
        self.num_classes = len(self.class_names)

        # Collect file paths per class
        for cls in self.class_names:
            cls_dir = self.data_dir / cls
            files = sorted([
                f for f in os.listdir(cls_dir)
                if os.path.isfile(cls_dir / f)
            ])
            self.file_paths[cls] = [cls_dir / f for f in files]
            self.class_counts[cls] = len(files)

        print(f"✅ Discovered {self.num_classes} classes in {self.data_dir}")
        for cls, count in sorted(self.class_counts.items()):
            print(f"   • {cls}: {count} images")

    def investigate(self) -> Dict:
        """
        Phase 1: Complete dataset investigation.
        Returns a comprehensive dataset summary dictionary.
        """
        print("\n" + "="*70)
        print("📊 PHASE 1: DATASET INVESTIGATION")
        print("="*70)

        summary = {}
        summary["total_images"] = sum(self.class_counts.values())
        summary["total_classes"] = self.num_classes
        summary["class_names"] = self.class_names
        summary["class_counts"] = self.class_counts

        # Identify healthy vs diseased classes
        healthy_classes = []
        diseased_classes = []
        for cls in self.class_names:
            if any(kw.lower() in cls.lower() for kw in HEALTHY_KEYWORDS):
                healthy_classes.append(cls)
            else:
                diseased_classes.append(cls)

        summary["healthy_classes"] = healthy_classes
        summary["diseased_classes"] = diseased_classes
        summary["healthy_count"] = sum(self.class_counts[c] for c in healthy_classes)
        summary["diseased_count"] = sum(self.class_counts[c] for c in diseased_classes)

        # Analyze image formats
        print("\n🔍 Analyzing image formats...")
        formats = Counter()
        dimensions = []
        corrupted = []
        duplicates = []

        seen_hashes = {}

        for cls in self.class_names:
            for fpath in tqdm(self.file_paths[cls], desc=f"  {cls[:30]:<30}"):
                try:
                    with Image.open(fpath) as img:
                        img.load()
                        fmt = img.format or "Unknown"
                        formats[fmt] += 1
                        dims = img.size
                        dimensions.append((dims[0], dims[1], cls))

                        # Check for duplicates via hash
                        with open(fpath, "rb") as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()
                        if file_hash in seen_hashes:
                            duplicates.append((fpath.name, seen_hashes[file_hash]))
                        else:
                            seen_hashes[file_hash] = fpath.name

                except Exception as e:
                    corrupted.append((str(fpath), str(e)))
                    print(f"   ⚠️  Corrupted: {fpath.name} - {e}")

        summary["image_formats"] = dict(formats)
        summary["corrupted_images"] = corrupted
        summary["duplicate_images"] = duplicates
        summary["total_corrupted"] = len(corrupted)
        summary["total_duplicates"] = len(duplicates)

        # Dimension statistics
        widths = [d[0] for d in dimensions]
        heights = [d[1] for d in dimensions]
        summary["width_stats"] = {
            "min": min(widths) if widths else 0,
            "max": max(widths) if widths else 0,
            "mean": np.mean(widths) if widths else 0,
            "std": np.std(widths) if widths else 0,
        }
        summary["height_stats"] = {
            "min": min(heights) if heights else 0,
            "max": max(heights) if heights else 0,
            "mean": np.mean(heights) if heights else 0,
            "std": np.std(heights) if heights else 0,
        }

        summary["unique_dimensions"] = len(set(zip(widths, heights)))

        # Dataset size
        total_size = sum(
            os.path.getsize(f)
            for files in self.file_paths.values()
            for f in files
        )
        summary["dataset_size_bytes"] = total_size
        summary["dataset_size_mb"] = total_size / (1024 * 1024)

        self.dataset_summary = summary
        self._print_summary(summary)
        return summary

    def _print_summary(self, summary: Dict) -> None:
        """Print the dataset investigation summary in a readable format."""
        print("\n" + "─"*70)
        print("📋 DATASET INVESTIGATION SUMMARY")
        print("─"*70)
        print(f"  Dataset Location  : {self.data_dir}")
        print(f"  Total Images      : {summary['total_images']:,}")
        print(f"  Total Classes     : {summary['total_classes']}")
        print(f"  Healthy Classes   : {len(summary['healthy_classes'])} → {summary['healthy_classes']}")
        print(f"  Diseased Classes  : {len(summary['diseased_classes'])} → {summary['diseased_classes']}")
        print(f"  Healthy Images    : {summary['healthy_count']:,}")
        print(f"  Diseased Images   : {summary['diseased_count']:,}")
        print(f"  Image Formats     : {summary['image_formats']}")
        print(f"  Dataset Size      : {summary['dataset_size_mb']:.2f} MB")
        print(f"  Width Range       : {summary['width_stats']['min']}–{summary['width_stats']['max']} px")
        print(f"  Height Range      : {summary['height_stats']['min']}–{summary['height_stats']['max']} px")
        print(f"  Corrupted Images  : {summary['total_corrupted']}")
        print(f"  Duplicate Images  : {summary['total_duplicates']}")
        print("─"*70)

    def get_class_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with class names and counts."""
        data = []
        for cls, count in sorted(self.class_counts.items()):
            is_healthy = any(kw.lower() in cls.lower() for kw in HEALTHY_KEYWORDS)
            plant = cls.split("___")[0] if "___" in cls else cls.split("_")[0]
            disease = cls.replace(f"{plant}___", "") if "___" in cls else "unknown"
            data.append({
                "class_name": cls,
                "count": count,
                "plant": plant,
                "disease": disease if not is_healthy else "healthy",
                "is_healthy": is_healthy,
            })
        df = pd.DataFrame(data)
        df = df.sort_values("count", ascending=False).reset_index(drop=True)
        return df

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a professional dataset investigation report as markdown.
        """
        if not self.dataset_summary:
            self.investigate()

        s = self.dataset_summary
        lines = []
        lines.append("# Dataset Investigation Report\n")
        lines.append(f"**Date:** {pd.Timestamp.now().strftime('%B %d, %Y')}\n")
        lines.append(f"**Dataset:** PlantVillage Leaf Disease Dataset\n")
        lines.append("---\n")
        lines.append("## 1. Dataset Overview\n")
        lines.append(f"- **Source:** PlantVillage Dataset\n")
        lines.append(f"- **Total Images:** {s['total_images']:,}\n")
        lines.append(f"- **Total Classes:** {s['total_classes']}\n")
        lines.append(f"- **Dataset Size:** {s['dataset_size_mb']:.2f} MB\n")
        lines.append(f"- **Image Format:** {list(s['image_formats'].keys())[0] if s['image_formats'] else 'N/A'}\n")
        lines.append(f"- **Image Dimensions:** {s['width_stats']['min']}×{s['height_stats']['min']} to {s['width_stats']['max']}×{s['height_stats']['max']} px\n")

        lines.append("\n## 2. Class Distribution\n\n")
        lines.append("| # | Class Name | Count | Plant | Type |\n")
        lines.append("|---|------------|-------|-------|------|\n")
        df = self.get_class_dataframe()
        for i, row in df.iterrows():
            dtype = "✅ Healthy" if row["is_healthy"] else "⚠️ Diseased"
            lines.append(f"| {i+1} | {row['class_name']} | {row['count']} | {row['plant']} | {dtype} |\n")

        lines.append("\n## 3. Quality Assessment\n")
        lines.append(f"- **Corrupted Images Found:** {s['total_corrupted']}\n")
        lines.append(f"- **Duplicate Images Found:** {s['total_duplicates']}\n")
        lines.append(f"- **Unique Image Dimensions:** {s['unique_dimensions']}\n")

        if s['corrupted_images']:
            lines.append("\n### Corrupted Files\n")
            for fpath, err in s['corrupted_images'][:20]:
                lines.append(f"- `{fpath}`: {err}\n")

        if s['duplicate_images']:
            lines.append("\n### Duplicate Files\n")
            for dup, orig in s['duplicate_images'][:20]:
                lines.append(f"- `{dup}` duplicates `{orig}`\n")

        report = "".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report)
            print(f"✅ Report saved to {output_path}")

        return report

    def get_clean_file_paths(self) -> Dict[str, List[Path]]:
        """
        Return file paths with corrupted and duplicate files removed.
        """
        # Re-investigate if needed
        if not self.dataset_summary:
            self.investigate()

        clean = {}
        # Build set of corrupted file path strings
        corrupted_set = set()
        for item in self.dataset_summary["corrupted_images"]:
            if isinstance(item, tuple):
                corrupted_set.add(item[0])
            elif isinstance(item, dict):
                corrupted_set.add(item.get("file", ""))

        # Build set of duplicate filenames (keep originals, remove duplicates)
        duplicate_names = set()
        for item in self.dataset_summary["duplicate_images"]:
            if isinstance(item, tuple):
                duplicate_names.add(item[0])
            elif isinstance(item, dict):
                duplicate_names.add(item.get("duplicate", ""))

        for cls in self.class_names:
            clean[cls] = [
                f for f in self.file_paths[cls]
                if str(f) not in corrupted_set and f.name not in duplicate_names
            ]

        return clean
