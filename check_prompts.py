import json
import re
import argparse
from pathlib import Path

CANDIDATE_KEYS = ["prompt", "input", "instruction", "question", "query", "source"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_prompt(obj):
    for k in CANDIDATE_KEYS:
        if k in obj and isinstance(obj[k], str):
            return norm(obj[k])

    msgs = obj.get("messages")
    if isinstance(msgs, list):
        user_parts = [
            m.get("content", "")
            for m in msgs
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_parts:
            return norm("\n".join(user_parts))

    for k in ("chosen", "rejected"):
        v = obj.get(k)
        if isinstance(v, dict):
            p = extract_prompt(v)
            if p:
                return p
    return None


def iter_arrow_rows(arrow_path: Path):
    try:
        from datasets import Dataset

        ds = Dataset.from_file(str(arrow_path))
        for row in ds:
            yield row
        return
    except Exception:
        pass

    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as e:
        raise ImportError(
            "Reading .arrow files requires `datasets` or `pyarrow`."
        ) from e

    with pa.memory_map(str(arrow_path), "r") as source:
        table = None
        try:
            reader = ipc.open_stream(source)
            table = reader.read_all()
        except Exception:
            source.seek(0)
            reader = ipc.open_file(source)
            table = reader.read_all()

    for row in table.to_pylist():
        if isinstance(row, dict):
            yield row


def load_prompts(path):
    prompts = []
    path = Path(path)

    if path.is_dir():
        # Hugging Face save_to_disk dataset directory
        try:
            from datasets import load_from_disk, DatasetDict
        except ImportError as e:
            raise ImportError(
                "Reading dataset directories requires `datasets`. "
                "Install with: pip install datasets"
            ) from e

        try:
            ds = load_from_disk(str(path))
            if isinstance(ds, DatasetDict):
                rows = ds["train"] if "train" in ds else next(iter(ds.values()))
            else:
                rows = ds
            for row in rows:
                p = extract_prompt(row)
                if p:
                    prompts.append(p)
        except Exception:
            # Fallback for directories that only contain raw/cached arrow shards.
            data_files = sorted(path.glob("data-*.arrow"))
            arrow_files = data_files if data_files else sorted(path.glob("*.arrow"))
            if not arrow_files:
                raise
            for af in arrow_files:
                for row in iter_arrow_rows(af):
                    p = extract_prompt(row)
                    if p:
                        prompts.append(p)
    elif path.suffix == ".arrow":
        # Single Arrow file, e.g. data-00000-of-00001.arrow
        for row in iter_arrow_rows(path):
            p = extract_prompt(row)
            if p:
                prompts.append(p)
    elif path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = extract_prompt(json.loads(line))
            if p:
                prompts.append(p)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            p = extract_prompt(row)
            if p:
                prompts.append(p)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dpo",
        required=True,
        help="Path to DPO dataset (.jsonl/.json/.arrow or HF dataset dir)",
    )
    ap.add_argument(
        "--rlvr",
        required=True,
        help="Path to RLVR dataset (.jsonl/.json/.arrow or HF dataset dir)",
    )
    ap.add_argument("--name", default="dataset", help="Display name")
    args = ap.parse_args()

    dpo = load_prompts(args.dpo)
    rlvr = load_prompts(args.rlvr)

    dset, rset = set(dpo), set(rlvr)
    same_order = dpo == rlvr
    same_set = dset == rset

    print(f"\n== {args.name} ==")
    print(f"DPO prompts:  {len(dpo)} (unique {len(dset)})")
    print(f"RLVR prompts: {len(rlvr)} (unique {len(rset)})")
    print(f"Same exact order: {same_order}")
    print(f"Same unique set:  {same_set}")

    only_dpo = list(dset - rset)
    only_rlvr = list(rset - dset)
    print(f"Only in DPO:  {len(only_dpo)}")
    print(f"Only in RLVR: {len(only_rlvr)}")

    if only_dpo:
        print("\nSample only-in-DPO:")
        for x in only_dpo[:3]:
            print("-", x[:200])
    if only_rlvr:
        print("\nSample only-in-RLVR:")
        for x in only_rlvr[:3]:
            print("-", x[:200])


if __name__ == "__main__":
    main()
