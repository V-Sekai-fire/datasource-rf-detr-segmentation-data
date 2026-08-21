# rf-detr-segmentation-data

Train/test split for RF-DETR instance-segmentation finetuning
([rf-detr-cpp](https://github.com/weftspun/rf-detr-cpp)), sourced from the
Roboflow Universe project
[chibifire/clothing-instance-segmentation-1f7c9](https://universe.roboflow.com/chibifire/clothing-instance-segmentation-1f7c9)
(version 2 -- the base export, no tiling/augmentation multiplication: 1158
source images). License: CC BY 4.0 (see `LICENSE.txt`).

Same source images as
[rf-detr-detection-data](https://github.com/weftspun/rf-detr-detection-data)
(separate repo per this project's modality convention), with the added
`segmentation_json` column.

## Contents

- `train.parquet` -- 696 images (Roboflow's own "train" split), zstd-compressed.
- `test.parquet` -- 462 images (Roboflow's "valid" + "test" splits merged,
  for more held-out data than either alone), zstd-compressed.
- `categories.json` -- the 47 clothing category id -> name mapping.
- `gen_split.py` -- the script that produced these files from a Roboflow
  COCO export (kept for reproducibility; requires a Roboflow API key to
  re-run against a newer project version).

## Schema (per row)

| column              | type              | meaning                                                                 |
| ------------------- | ----------------- | ----------------------------------------------------------------------- |
| `image_id`          | int64             | unique id (namespaced by source split)                                  |
| `file_name`         | string            | original file name                                                      |
| `width`, `height`   | int32             | image dimensions (px)                                                   |
| `image_bytes`       | binary            | the JPEG file, byte-identical to the Roboflow export                    |
| `category_id`       | list<int64>       | one entry per annotation instance                                       |
| `bbox_xywh`         | list<list<float>> | COCO-native `[x, y, w, h]`, absolute pixels                             |
| `segmentation_json` | list<string>      | one JSON-encoded COCO segmentation per instance (polygon list-of-lists) |

Images are already square-letterboxed to 432x432 by Roboflow's own
preprocessing (auto-orient + "Fit (black edges) in" resize) -- that
transform happened upstream of this repo, not applied here.

## Loading

```python
import json
import pyarrow.parquet as pq
table = pq.read_table("train.parquet")
row = table.to_pylist()[0]
polygons = [json.loads(s) for s in row["segmentation_json"]]
```

Consumed by `rf-detr-cpp`'s dataset loader via
`gen_reference/gen_from_parquet_split.py` (converts each split into the
existing `write_arr`-based `.bin` format `src/dataset.cpp` reads, decoding
masks the same way `ConvertCoco(include_masks=True)` does upstream).
