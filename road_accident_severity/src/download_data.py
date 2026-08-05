"""Download Israel CBS road-accident PUF (2020–2024) via data.gov.il datastore API."""
from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"

RESOURCES = {
    2024: "05d14adb-fe54-49f7-b7ce-f30348e2d959",
    2023: "ae0f1679-139f-4e69-a869-a60d5d76518b",
    2022: "ede3f02a-f9aa-4a6f-9eca-4c11a06f0043",
    2021: "6957e7c2-6d68-4332-bbc8-2ee8d5ba6bd6",
    2020: "70a93f04-2ffe-4b02-a062-a818600a5b67",
}
DICT_RID = "c557fe0c-5f18-41ff-b756-44d26ed4aee4"
PACKAGE = "https://data.gov.il/he/datasets/lamas/2023-puf"


def _fetch_all(rid: str, page: int = 32000) -> tuple[list[str], list[dict]]:
    records: list[dict] = []
    offset = 0
    fields: list[str] | None = None
    while True:
        q = urllib.parse.urlencode({"resource_id": rid, "limit": page, "offset": offset})
        url = f"https://data.gov.il/api/3/action/datastore_search?{q}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "road-accident-severity/1.0 (course project)"}
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            payload = json.load(r)
        res = payload["result"]
        if fields is None:
            fields = [f["id"] for f in res["fields"] if f["id"] != "_id"]
        batch = res["records"]
        records.extend(batch)
        total = int(res.get("total", 0))
        print(f"  offset={offset} batch={len(batch)} so_far={len(records)}/{total}")
        if not batch or len(records) >= total:
            break
        offset += len(batch)
        time.sleep(0.15)
    assert fields is not None
    return fields, records


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    print("Downloading codebook dictionary…")
    dfields, drecs = _fetch_all(DICT_RID)
    dict_path = DATA / "codebook_dictionary_2024.csv"
    with dict_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=dfields)
        w.writeheader()
        for rec in drecs:
            w.writerow({k: rec.get(k) for k in dfields})
    print(" wrote", dict_path, len(drecs))

    common: list[str] | None = None
    all_rows: list[dict] = []
    for year, rid in RESOURCES.items():
        print(f"YEAR {year}")
        fields, recs = _fetch_all(rid)
        if common is None:
            common = fields
        year_path = RAW / f"accidents_{year}.csv"
        with year_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=common)
            w.writeheader()
            for rec in recs:
                w.writerow({k: rec.get(k) for k in common})
        print(" wrote", year_path, len(recs))
        for rec in recs:
            row = {k: rec.get(k) for k in common}
            row["year_source"] = year
            all_rows.append(row)

    assert common is not None
    comb = DATA / "accidents_2020_2024.csv"
    cols = list(common) + ["year_source"]
    with comb.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"COMBINED {len(all_rows)} rows → {comb}")
    print(f"Source package: {PACKAGE}")


if __name__ == "__main__":
    main()
