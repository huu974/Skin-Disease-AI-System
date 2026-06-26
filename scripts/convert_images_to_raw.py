"""Convert an ImageFolder-style training dataset to binary .raw images.

The default input matches this project's training dataset:
    ./skin diseases/train-new
The default output is a separate directory in the project root:
    ./train-new-raw

Each image is decoded with Pillow, converted to a fixed pixel mode, and written
as raw pixel bytes. Directory structure is preserved under the output folder.
Metadata needed to read the raw files is written to a CSV manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".gif",
    ".jfif",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}

MODE_CHANNELS = {
    "L": 1,
    "RGB": 3,
    "RGBA": 4,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None

    normalized = value.lower().replace(" ", "")
    if "x" not in normalized:
        raise argparse.ArgumentTypeError("resize must use WIDTHxHEIGHT, for example 300x300")

    width_text, height_text = normalized.split("x", 1)
    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resize width and height must be integers") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resize width and height must be positive")

    return width, height


def iter_image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def output_path_for(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = input_path.relative_to(input_dir)
    return output_dir / relative_path.with_suffix(".raw")


def image_to_array_bytes(image: Image.Image, layout: str) -> bytes:
    if layout == "hwc":
        return image.tobytes()

    if layout == "chw":
        channels = image.split()
        return b"".join(channel.tobytes() for channel in channels)

    raise ValueError(f"Unsupported layout: {layout}")


def convert_one(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
    mode: str,
    resize: tuple[int, int] | None,
    layout: str,
    overwrite: bool,
) -> dict[str, str | int]:
    raw_path = output_path_for(input_path, input_dir, output_dir)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {raw_path}")

    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert(mode)
        if resize is not None:
            image = image.resize(resize, Image.Resampling.BILINEAR)

        width, height = image.size
        raw_bytes = image_to_array_bytes(image, layout)
        raw_path.write_bytes(raw_bytes)

    channels = MODE_CHANNELS[mode]
    expected_bytes = width * height * channels
    return {
        "source_path": input_path.as_posix(),
        "raw_path": raw_path.as_posix(),
        "class_name": input_path.parent.relative_to(input_dir).as_posix(),
        "width": width,
        "height": height,
        "channels": channels,
        "mode": mode,
        "layout": layout.upper(),
        "dtype": "uint8",
        "byte_count": len(raw_bytes),
        "expected_byte_count": expected_bytes,
    }


def write_manifest(manifest_path: Path, rows: list[dict[str, str | int]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "raw_path",
        "class_name",
        "width",
        "height",
        "channels",
        "mode",
        "layout",
        "dtype",
        "byte_count",
        "expected_byte_count",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str | int]], failed_count: int) -> dict[str, int | float]:
    total_bytes = sum(int(row["byte_count"]) for row in rows)
    return {
        "converted_count": len(rows),
        "failed_count": failed_count,
        "total_byte_count": total_bytes,
        "total_mib": round(total_bytes / (1024 * 1024), 4),
    }


def write_summary(summary_path: Path, summary: dict[str, int | float]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert all training images to binary .raw pixel files.",
    )
    parser.add_argument(
        "--input-dir",
        default=project_path("skin diseases/train-new"),
        type=Path,
        help="ImageFolder-style training dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=project_path("train-new-raw"),
        type=Path,
        help="Directory where .raw files will be written.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        type=Path,
        help="CSV metadata output path. Defaults to OUTPUT_DIR/manifest.csv.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        type=Path,
        help="JSON summary output path. Defaults to OUTPUT_DIR/summary.json.",
    )
    parser.add_argument(
        "--mode",
        default="RGB",
        choices=sorted(MODE_CHANNELS),
        help="Pixel mode for raw output.",
    )
    parser.add_argument(
        "--resize",
        default=None,
        type=parse_size,
        help="Optional resize before writing raw bytes, e.g. 300x300.",
    )
    parser.add_argument(
        "--layout",
        default="hwc",
        choices=("hwc", "chw"),
        help="Byte layout: HWC interleaved pixels or CHW planar channels.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .raw files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    manifest_path = args.manifest or output_dir / "manifest.csv"
    summary_path = args.summary or output_dir / "summary.json"

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}", file=sys.stderr)
        return 1

    image_files = iter_image_files(input_dir)
    if not image_files:
        print(f"No supported image files found under: {input_dir}", file=sys.stderr)
        return 1

    rows = []
    failed = []
    total = len(image_files)

    for index, image_path in enumerate(image_files, start=1):
        try:
            rows.append(
                convert_one(
                    input_path=image_path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    mode=args.mode,
                    resize=args.resize,
                    layout=args.layout,
                    overwrite=args.overwrite,
                )
            )
        except (OSError, UnidentifiedImageError, FileExistsError) as exc:
            failed.append((image_path, exc))
            print(f"[{index}/{total}] failed: {image_path} ({exc})", file=sys.stderr)
        else:
            if index % 100 == 0 or index == total:
                print(f"[{index}/{total}] converted: {image_path}")

    if rows:
        write_manifest(manifest_path, rows)
    summary = build_summary(rows, len(failed))
    write_summary(summary_path, summary)

    print(f"Converted: {len(rows)}")
    print(f"Failed: {len(failed)}")
    print(f"Total bytes: {summary['total_byte_count']} ({summary['total_mib']} MiB)")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
