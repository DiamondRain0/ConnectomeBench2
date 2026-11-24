"""smoke_test_v2_packaging.py — offline test of the v2 packaging chain (no CAVE, no rendering).

Fabricates renderer-style sub-archives (questions.parquet + .npy geometry +
EM / split-mask PNGs, using the renderer's naming and encodings), then runs

    build_giga_parquet.py → postprocess_giga_parquet.py → build_parquet_dataset.py

and checks that the shards (a) have exactly the column names / types of the
released v2 dataset and (b) load through the training data layer with the
right tasks, labels, views and split-mask targets.

    python scripts/data_generation/neurips/smoke_test_v2_packaging.py [--reference demo.parquet] [--keep]

`--reference` (optional) is the released v2 `demo.parquet`; when given, the
schema is compared column-by-column against it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
NEURIPS = ROOT / "scripts" / "data_generation" / "neurips"
sys.path.insert(0, str(ROOT / "src" / "data_generation"))
sys.path.insert(0, str(ROOT / "src" / "neurips_training"))


def _geom(rng, dual: bool) -> np.ndarray:
    g = np.zeros((3, 7, 224, 224), np.float16)
    g[:, 0, 60:160, 60:160] = 1
    g[:, 1] = rng.random((3, 224, 224)).astype(np.float16) * g[:, 0]
    g[:, 2:5] = (rng.random((3, 3, 224, 224)) * 2 - 1).astype(np.float16)
    g[:, 5, 60:160, 60:110] = 1
    if dual:
        g[:, 6, 60:160, 110:160] = 1
    else:
        g[:, 5, 60:160, 60:160] = 1
    return g


def _em(path: Path, rng, dual: bool) -> None:
    a = np.zeros((224, 224, 3), np.uint8)
    a[..., 0] = rng.integers(0, 255, (224, 224))
    a[80:150, 60:110, 1] = 255
    if dual:
        a[80:150, 110:160, 2] = 255
    Image.fromarray(a, "RGB").save(path)


def _mask(path: Path) -> None:  # renderer encoding: 0 = A, 128 = B, 255 = bg
    a = np.full((224, 224), 255, np.uint8)
    a[60:160, 60:110] = 0
    a[60:160, 110:160] = 128
    Image.fromarray(a, "L").save(path)


def make_renderer_output(root: Path, rng) -> None:
    samples = {
        "edits_and_adj_controls_mouse": [
            ("op_merge", {"source_type": "operation", "is_merge": True, "operation_id": "11",
                          "before_root_ids": [1, 2], "after_root_ids": [3]}, ["geometry", "geometry_single"], False),
            ("op_split", {"source_type": "operation", "is_merge": False, "operation_id": "12",
                          "before_root_ids": [4], "after_root_ids": [5, 6]},
             ["geometry", "geometry_single", "split_mask_front", "split_mask_side", "split_mask_top"], True),
            ("inv_split", {"source_type": "control", "strategy": "inversion", "is_merge": True,
                           "operation_id": "13", "before_root_ids": [7, 8], "after_root_ids": [9]},
             ["geometry", "geometry_single", "split_mask_front", "split_mask_side", "split_mask_top"], True),
            ("adj", {"source_type": "control", "strategy": "adjacent_in_cutout", "is_merge": True,
                     "operation_id": "11", "adjacent_root": 2, "base_root": 1}, ["geometry"], False),
            ("no_single", {"source_type": "operation", "is_merge": False, "operation_id": "14",
                           "before_root_ids": [4], "after_root_ids": [5, 6]}, ["geometry"], True),  # dropped
        ],
        "junction_controls_mouse": [
            ("jn", {"source_type": "control", "operation_id": "-5", "root_ids": [10]}, ["geometry"], False),
        ],
        "synapse_controls_fly": [
            ("syn", {"source_type": "synapse_pair", "operation_id": "77", "root_ids": [20, 21],
                     "synapse_ctr_pt_nm": [100.0, 100.0, 100.0]}, ["geometry"], False),
        ],
    }
    for k, (archive, rows) in enumerate(samples.items()):
        d = root / archive
        (d / "images").mkdir(parents=True)
        recs = []
        for j, (name, meta, gtypes, _) in enumerate(rows):
            species = archive.rsplit("_", 1)[1]
            meta = {**meta, "species": species, "interface_point_nm": [1000.0 + 90000 * j, 2000.0, 3000.0],
                    "edit_point_nm": [1100.0 + 90000 * j, 2000.0, 3000.0], "view_extent_nm": 5000.0}
            images, itypes = [], []
            for t in gtypes:
                if t.startswith("geometry"):
                    p = d / "images" / f"{name}_{t}.npy"
                    np.save(p, _geom(rng, dual=(t == "geometry" and name not in ("jn",))))
                else:
                    p = d / "images" / f"{name}_{t}.png"
                    _mask(p)
                images.append(f"images/{p.name}")
                itypes.append(t)
            for v in ("xy", "xz", "yz", "best"):
                p = d / "images" / f"{name}_em_{v}.png"
                _em(p, rng, dual=name not in ("jn",))
                images.append(f"images/{p.name}")
                itypes.append(f"em_{v}")
            meta["image_types"] = itypes
            recs.append({"question_type": "x", "answer": "True", "images": images,
                         "metadata": meta, "sample_hash": f"{k}{j:011d}"})
        pq.write_table(pa.Table.from_pylist(recs), d / "questions.parquet")


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=None, help="released v2 demo.parquet")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    from connectome.cube_splits_v2 import CubeTable
    work = Path(tempfile.mkdtemp(prefix="cb2_pack_"))
    try:
        rng = np.random.default_rng(0)
        renders = work / "renders"
        make_renderer_output(renders, rng)
        tables = work / "cube_tables"
        CubeTable("mouse", [0.0, 0.0, 0.0], cubes={(0, 0, 0): "train", (1, 0, 0): "val",
                                                    (2, 0, 0): "test"}).save(tables / "mouse.json")
        CubeTable("fly", [0.0, 0.0, 0.0], cubes={(0, 0, 0): "train"}).save(tables / "fly.json")

        py = sys.executable
        run([py, NEURIPS / "build_giga_parquet.py", "--out-root", renders])
        run([py, NEURIPS / "postprocess_giga_parquet.py", "--parquet", renders / "combined.parquet",
             "--splits-root", tables])
        run([py, NEURIPS / "postprocess_giga_parquet.py", "--parquet", renders / "combined.parquet",
             "--splits-root", tables, "--dry-run"])  # idempotency: must keep every row
        out = work / "hf"
        run([py, NEURIPS / "build_parquet_dataset.py", "--parquet", renders / "combined.parquet",
             "--out-root", out, "--workers", "1", "--read-threads", "2", "--demo-per-type", "1"])

        # ── checks ──
        shards = sorted(out.glob("*/*-*.parquet"))
        t = pa.concat_tables([pq.read_table(s) for s in shards])
        df = t.to_pandas()
        print(f"\npacked {len(df)} rows in {len(shards)} shards: "
              f"{df.groupby(['sample_type', 'split']).size().to_dict()}")
        assert set(df.sample_type) == {"merge_edit", "split_edit", "adjacent_control",
                                       "junction_control", "synapse_control"}, set(df.sample_type)
        assert (df.sample_type == "split_edit").sum() == 2, "op_split + inv_split expected; no_single dropped"
        assert df.source_archive.str.match(r"^unified_(controls_)?(mouse|fly)$").all()
        se = df[df.sample_type == "split_edit"].iloc[0]
        assert list(se.task_routing) == ["false_merge_identification", "split_mask_generation"]
        assert se.has_after_mask and np.isnan(se.false_split_correction_label)
        meta = json.loads(se.metadata)
        assert meta["strategy"] == "split_correction" and meta["is_merge"] is False
        assert len(meta["before_root_ids"]) == 1 and len(meta["after_root_ids"]) == 2
        m = np.asarray(Image.open(__import__("io").BytesIO(se.split_mask_front["bytes"])))
        assert set(np.unique(m)) == {0, 1, 255}, np.unique(m)
        g = np.load(__import__("io").BytesIO(se.geometry))["arr_0"]
        assert g.shape == (3, 7, 224, 224) and g.dtype == np.float16 and g[:, 6].max() == 0, "single render"
        assert t.schema.field("em_xy_before").type == pa.struct([("bytes", pa.binary()), ("path", pa.string())])

        if args.reference is not None:
            ref = pq.read_schema(args.reference)
            ours = t.schema
            assert ref.names == ours.names, f"\nref : {ref.names}\nours: {ours.names}"
            for f in ref:
                rt, ot = f.type, ours.field(f.name).type
                same = rt == ot or (pa.types.is_struct(rt) and pa.types.is_struct(ot)
                                    and [x.name for x in rt] == [x.name for x in ot])
                assert same, f"{f.name}: ref {rt} vs ours {ot}"
            print(f"schema matches reference {args.reference.name} ({len(ref.names)} columns)")

        from data import NeurIPSDataset, BlendConfig
        n_items = 0
        for split in ("train", "val", "test"):
            if not (out / split).exists():
                continue
            ds = NeurIPSDataset(out, split, blend=BlendConfig(), sample_all=True)
            assert ds.schema == "v2"
            for it in ds:
                n_items += 1
                if it["has_mask_gt"]:
                    assert set(it["mask_label"].unique().tolist()) <= {0, 1, 2}
                    assert (it["mask_label"] == 2).any()
        print(f"training loader read {n_items} view items")
        print("\nV2 PACKAGING SMOKE TEST PASSED")
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept {work}")


if __name__ == "__main__":
    main()
