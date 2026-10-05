# datasource-rf-detr-segmentation-data

Train and test splits of a clothing instance-segmentation dataset as Parquet, for fine-tuning RF-DETR's segmentation head.

## What it is for

Each row holds one image with a COCO-style box and polygon mask for every clothing instance in it, so a fine-tune reads the split without the original export. `datasource-rf-detr-detection-data` carries the same images with boxes alone. `LICENSE.txt` names the source project and its attribution.

## Build and run

The splits open with any Parquet reader. `gen_split.py` rebuilds them from a COCO export of the source project:

```sh
python gen_split.py segmentation /path/to/coco-export
```

## Licence

CC BY 4.0 for the data and for this repository's additions; see `LICENSE.txt`.
