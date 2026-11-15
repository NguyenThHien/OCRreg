"""Chuẩn bị crop/đáp án; không chạy benchmark hoặc cập nhật model."""

from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm

import config
from analysis.metrics import normalize_text
from utils import OCRError, VietnameseParser, load_model_config, run_cli, save_json, write_csv

FIELDS = ["crop", "ground_truth", "source_image", "line", "width", "height"]


# Đọc tám tọa độ đầu dòng, giữ nguyên dấu phẩy/khoảng trắng trong phần nhãn.
def read_regions(path):
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split(",", 8)
        try:
            if len(fields) != 9:
                raise ValueError
            points = np.array(fields[:8], dtype=np.float32).reshape(4, 2)
            if not np.isfinite(points).all():
                raise ValueError
        except ValueError as error:
            raise OCRError(f"Nhãn sai tọa độ/định dạng: {path}, dòng {number}.") from error
        yield number, normalize_text(fields[8]), points


# Chỉnh phối cảnh tứ giác lồi; trả về None khi vùng không thể dùng để nhận dạng.
def crop_region(image, points):
    top_left, top_right, bottom_right, bottom_left = points
    width = round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    height = round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
    if width < 2 or height < 2 or not cv2.isContourConvex(points):
        return None
    target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(image, matrix, (width, height))


# Xác định lý do loại; chỉ train/val kiểm tra vocab và giới hạn độ dài nhãn.
def exclusion_reason(text, vocabulary=None):
    if text.strip() == "###":
        return "Nhãn bỏ qua (###)"
    if not text.strip():
        return "Nhãn rỗng"
    if vocabulary is not None and set(text) - vocabulary:
        return "Ký tự ngoài vocab"
    if vocabulary is not None and len(text) > config.MAX_TEXT_LENGTH:
        return "Nhãn vượt giới hạn giải mã"
    return ""


# Tạo crop và samples.csv cho một tập; ảnh gốc không bị thay đổi.
def prepare_split(source, output, split, vocabulary=None):
    images = sorted((source / config.SOURCE_FOLDERS[split]).glob("*.jpg"))
    if not images:
        raise OCRError(f"Không tìm thấy ảnh của tập {split}.")
    if output.exists() and any(output.iterdir()):
        raise OCRError(f"Đã có dữ liệu tại {output}; không tạo lại để tránh trộn các lần chạy.")
    (output / "images").mkdir(parents=True, exist_ok=True)
    samples, excluded = [], []
    for path in tqdm(images, desc=f"Chuẩn bị {split}"):
        image_id = int(path.stem.removeprefix("im"))
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise OCRError(f"Không đọc được ảnh: {path}.")
        for number, text, points in read_regions(source / "labels" / f"gt_{image_id}.txt"):
            reason = exclusion_reason(text, vocabulary)
            crop = None if reason else crop_region(image, points)
            reason = reason or ("Tứ giác không hợp lệ" if crop is None else "")
            row = {"source_image": path.name, "line": number, "ground_truth": text}
            if reason:
                excluded.append({**row, "reason": reason})
                continue
            relative = f"images/{image_id:04d}_{number:04d}.png"
            success, encoded = cv2.imencode(".png", crop)
            if not success:
                raise OCRError(f"Không mã hóa được crop: {path}, dòng {number}.")
            encoded.tofile(output / relative)
            samples.append({**row, "crop": relative, "width": crop.shape[1], "height": crop.shape[0]})
    if not samples:
        raise OCRError(f"Tập {split} không có vùng chữ hợp lệ.")
    write_csv(output / "samples.csv", FIELDS, samples)
    write_csv(output / "excluded.csv", ["source_image", "line", "ground_truth", "reason"], excluded)
    summary = {
        "source_images": len(images), "accepted": len(samples), "excluded": len(excluded),
        "exclusion_reasons": dict(Counter(row["reason"] for row in excluded)),
    }
    save_json(output / "preparation.json", summary)
    print(f"{split}: giữ {len(samples)} crop, loại {len(excluded)} vùng.")
    return summary


# Kiểm tra đúng mã ảnh của ba tập gốc trước khi tạo dữ liệu cho thí nghiệm.
def check_source_splits():
    for split, (start, end) in config.SOURCE_RANGES.items():
        expected = {f"im{index:04d}.jpg" for index in range(start, end + 1)}
        actual = {path.name for path in (config.SOURCE / config.SOURCE_FOLDERS[split]).glob("*.jpg")}
        if actual != expected:
            raise OCRError(f"Tập {split} thiếu/trùng/sai mã ảnh. Cần các ảnh từ {start} đến {end}.")


# Chọn chuẩn bị test hoặc train/val; test luôn giữ cả nhãn ngoài vocab model.
def main():
    parser = VietnameseParser(description=__doc__)
    parser.add_argument("split", choices=["test", "train"])
    args = parser.parse_args()
    check_source_splits()
    if args.split == "test":
        prepare_split(config.SOURCE, config.CROPS / "test", "test")
    else:
        if not (config.BEFORE / "metrics.json").is_file():
            raise OCRError("Hãy benchmark và phân tích pretrained trước khi chuẩn bị train/val.")
        vocabulary = set(load_model_config(config.BEFORE / "model.yml")["vocab"])
        for split in ("train", "val"):
            prepare_split(config.SOURCE, config.CROPS / split, split, vocabulary)


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
