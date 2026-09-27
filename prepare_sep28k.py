"""Build a local experiment manifest from SEP-28k-Extended clip metadata."""
import argparse
from pathlib import Path

import pandas as pd

LABELS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True, help="SEP-28k-Extended_clips.csv")
    p.add_argument("--clips", type=Path, required=True, help="root of extracted clips")
    p.add_argument("--output", type=Path, required=True, help="new manifest.csv")
    p.add_argument("--split-column", default="SEP28k-E")
    p.add_argument("--group-by", choices=["speaker", "episode", "show"], default="speaker")
    p.add_argument("--allow-missing", action="store_true", help="explicitly permit missing clip files")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; choose a new path")
    df = pd.read_csv(args.metadata, dtype={"EpId": str, "ClipId": str, "Show": str})
    needed = ["Show", "EpId", "ClipId", args.split_column, *LABELS]
    if args.group_by == "speaker":
        needed.append("speaker")
    missing_columns = sorted(set(needed) - set(df.columns))
    if missing_columns:
        p.error(f"Missing metadata columns: {missing_columns}")
    split = df[args.split_column].astype("string").str.lower()
    if not split.dropna().isin(["train", "dev", "test"]).all():
        p.error("Selected split column contains values other than train/dev/test")
    df = df.loc[split.notna()].copy()
    df["split"] = split[split.notna()].to_numpy()
    if args.group_by == "speaker":
        if df.speaker.isna().any() or df.speaker.astype(str).str.strip().eq("").any():
            p.error("Missing speaker identities; use episode/show grouping and report that scope")
        df["group_id"] = df.speaker.astype(str).str.strip()
    elif args.group_by == "episode":
        df["group_id"] = df.Show + "/" + df.EpId
    else:
        df["group_id"] = df.Show
    if df.groupby("group_id").split.nunique().max() > 1:
        p.error("Selected group IDs cross split boundaries. Choose a compatible split or grouping")
    root = args.clips.resolve()
    # Matches the filename created by Apple's extract_clips.py.
    paths = [root / show / ep / f"{show}_{ep}_{clip}.wav"
             for show, ep, clip in zip(df.Show, df.EpId, df.ClipId)]
    exists = [path.is_file() for path in paths]
    if not all(exists) and not args.allow_missing:
        p.error(f"{len(exists)-sum(exists)} clips missing. Example: {next(path for path, ok in zip(paths, exists) if not ok)}")
    df["audio_path"] = [str(path) for path in paths]
    df = df.loc[exists].copy()
    if set(df.split) != {"train", "dev", "test"}:
        p.error("All three splits must have available clips")
    if df.duplicated(["audio_path"]).any():
        p.error("Duplicate clip rows in metadata")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df[["audio_path", "group_id", "split", *LABELS]].to_csv(args.output, index=False)
    print(f"Wrote {len(df)} clips to {args.output}; skipped {len(exists)-sum(exists)} missing files")
    print(df.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
