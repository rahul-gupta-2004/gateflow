from pathlib import Path
from detector import PlateDetector


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    detector = PlateDetector(languages=["en"])

    image_files = sorted(data_dir.glob("*.jpg"))
    if not image_files:
        print(f"No .jpg images found in {data_dir}")
        return

    for img_path in image_files:
        text, _ = detector.extract_text(img_path)
        status = text if text else "No plate detected"
        print(f"{img_path.name}: {status}")


if __name__ == "__main__":
    main()