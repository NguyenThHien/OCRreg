"""Tiện ích đọc/ghi và lỗi chung; không chứa công thức chỉ số hay vòng train."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import traceback

import yaml

import config


class OCRError(ValueError):
    """Lỗi đã có thông báo tiếng Việt để in cho người chạy."""


class VietnameseParser(argparse.ArgumentParser):
    # Thống nhất lỗi tham số CLI bằng tiếng Việt.
    def error(self, message):
        raise OCRError("Tham số không hợp lệ. Dùng --help để xem cách chạy.")


# Chạy một tác vụ, in lỗi tiếng Việt và giữ traceback kỹ thuật trong output/logs.
def run_cli(main):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
        return 0
    except KeyboardInterrupt:
        print("Đã dừng theo yêu cầu của bạn.", file=sys.stderr)
        return 130
    except Exception as error:
        message = str(error) if isinstance(error, OCRError) else "Không thể hoàn tất. Hãy xem nhật ký lỗi."
        print(f"Lỗi: {message}", file=sys.stderr)
        log = config.OUTPUT / "logs/errors.log"
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as file:
                file.write(traceback.format_exc() + "\n")
            print(f"Chi tiết: {log}", file=sys.stderr)
        except OSError:
            print("Không thể ghi nhật ký lỗi.", file=sys.stderr)
        return 1


# Đọc CSV và kiểm tra những cột mà bước gọi cần dùng.
def read_csv(path, required=()):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not set(required).issubset(reader.fieldnames or []):
            raise OCRError(f"CSV thiếu cột cần thiết: {path}.")
        rows = list(reader)
    if any(row.get(key) is None for row in rows for key in required):
        raise OCRError(f"CSV có dòng thiếu dữ liệu: {path}.")
    return rows


# Ghi CSV có BOM để Excel hiển thị đúng dấu tiếng Việt, kể cả bảng không có dòng.
def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# Lưu thông tin thực tế bằng JSON UTF-8, không chấp nhận NaN/Infinity.
def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


# Nhận diện chính xác nội dung ảnh hoặc trọng số, không chỉ tên tệp.
def file_sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


# Ghi môi trường thực tế để kiểm tra điều kiện đo trước/sau có tương đương không.
def runtime_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "torchvision", "vietocr", "numpy", "Pillow", "opencv-python-headless"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


# Nạp cấu hình gốc hoặc đã lưu; đường dẫn trọng số tương đối tính từ chính YAML.
def load_model_config(path=None, device="cpu"):
    if path is None:
        from vietocr.tool.config import Cfg

        result = dict(Cfg.load_config_from_name(config.MODEL_NAME))
        result["dataset"].update({
            "image_height": config.IMAGE_HEIGHT,
            "image_min_width": config.IMAGE_MIN_WIDTH,
            "image_max_width": config.IMAGE_MAX_WIDTH,
        })
    else:
        result = yaml.safe_load(path.read_text(encoding="utf-8"))
        location = result["weights"]
        if not location.startswith(("http://", "https://")):
            result["weights"] = str((path.parent / location).resolve())
    result["device"] = device
    result["cnn"]["pretrained"] = False
    result.setdefault("predictor", {})["beamsearch"] = False
    return result
