"""Benchmark toàn bộ test: before dùng pretrained, after dùng checkpoint tốt nhất."""

import json
import os
import shutil
from pathlib import Path
from time import perf_counter

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from vietocr.tool.predictor import Predictor
from vietocr.tool.translate import process_input, translate
from vietocr.tool.utils import download_weights

import config
from analysis.metrics import edit_distance, latency_metrics, normalize_text, text_metrics
from utils import (
    OCRError, VietnameseParser, file_sha256, load_model_config, read_csv,
    run_cli, runtime_versions, save_json, write_csv,
)


# Tiền xử lý, giải mã greedy và lấy chuỗi bằng đúng giới hạn đã khai báo trong config.
def predict_image(predictor, image):
    size = predictor.config["dataset"]
    tensor = process_input(image, size["image_height"], size["image_min_width"], size["image_max_width"])
    encoded, _ = translate(tensor.to(predictor.device), predictor.model, config.MAX_TEXT_LENGTH)
    return predictor.vocab.decode(encoded[0].tolist())


# Đọc toàn bộ test, chặn thiếu mẫu và trùng định danh; không lấy tập con.
def load_test_samples():
    path = config.CROPS / "test/samples.csv"
    rows = read_csv(path, ["crop", "ground_truth", "source_image", "line"])
    preparation = json.loads((path.parent / "preparation.json").read_text(encoding="utf-8"))
    identities = {(row["source_image"], row["line"]) for row in rows}
    if not rows or len(rows) != preparation["accepted"] or len(identities) != len(rows):
        raise OCRError("Bảng test thiếu mẫu hoặc trùng định danh. Hãy kiểm tra dữ liệu đã chuẩn bị.")
    for row in rows:
        if (not row["source_image"] or not row["line"] or not row["ground_truth"].strip()
                or not (path.parent / row["crop"]).is_file()):
            raise OCRError(f"Nhãn rỗng hoặc thiếu crop: {row['crop']}.")
    return rows


# Dự đoán từng crop và chỉ bấm giờ quanh predict, sau khi đọc ảnh/băm nội dung.
def predict_test(predictor, samples, output):
    rows, times = [], []
    for sample in tqdm(samples, desc="Benchmark test"):
        image_path = config.CROPS / "test" / sample["crop"]
        digest = file_sha256(image_path)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
        if predictor.device.startswith("cuda"):
            torch.cuda.synchronize(predictor.device)
        started = perf_counter()
        prediction = predict_image(predictor, rgb)
        if predictor.device.startswith("cuda"):
            torch.cuda.synchronize(predictor.device)
        elapsed = (perf_counter() - started) * 1000
        truth, prediction = normalize_text(sample["ground_truth"]), normalize_text(prediction)
        rows.append({
            "crop": Path(os.path.relpath(image_path, output)).as_posix(),
            "source_image": sample["source_image"], "line": sample["line"],
            "ground_truth": truth, "prediction": prediction, "crop_sha256": digest,
            "correct": truth == prediction, "edit_distance": edit_distance(truth, prediction),
            "predict_ms": elapsed,
        })
        times.append(elapsed)
    return rows, times


# Nạp model đúng giai đoạn, warmup rồi lưu dự đoán/chỉ số của toàn bộ test.
def main():
    parser = VietnameseParser(description=__doc__)
    parser.add_argument("phase", choices=["before", "after"])
    phase = parser.parse_args().phase
    output = config.BEFORE if phase == "before" else config.AFTER
    if output.exists() and any(output.iterdir()):
        raise OCRError(f"Đã có kết quả tại {output}; không ghi đè lần đo cũ.")
    if config.BENCHMARK_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise OCRError("CUDA chưa khả dụng. Hãy kiểm tra BENCHMARK_DEVICE trong config.py.")
    torch.set_num_threads(config.CPU_THREADS)
    torch.manual_seed(config.SEED)
    samples = load_test_samples()
    model_path = None if phase == "before" else config.CHECKPOINTS / "model.yml"
    model_config = load_model_config(model_path, config.BENCHMARK_DEVICE)
    weights = Path(download_weights(model_config["weights"]))
    predictor = Predictor({**model_config, "weights": str(weights)})
    predictor.model.eval()
    with torch.inference_mode():
        with Image.open(config.CROPS / "test" / samples[0]["crop"]) as image:
            for _ in range(config.WARMUP_RUNS):
                predict_image(predictor, image.convert("RGB"))
        rows, times = predict_test(predictor, samples, output)
    scores = text_metrics([row["ground_truth"] for row in rows], [row["prediction"] for row in rows])
    summary = {
        **scores, **latency_metrics(times), "phase": phase, "weights_sha256": file_sha256(weights),
        "device": config.BENCHMARK_DEVICE, "cpu_threads": config.CPU_THREADS,
        "warmup_runs": config.WARMUP_RUNS, "batch_size": 1, "environment": runtime_versions(),
        "max_text_length": config.MAX_TEXT_LENGTH, "seed": config.SEED,
        "samples_sha256": file_sha256(config.CROPS / "test/samples.csv"),
    }
    write_csv(output / "predictions.csv", list(rows[0]), rows)
    shutil.copyfile(weights, output / "weights.pth")
    model_config["weights"] = "weights.pth"
    (output / "model.yml").write_text(yaml.safe_dump(model_config, allow_unicode=True), encoding="utf-8")
    save_json(output / "metrics.json", summary)
    print(f"Exact Match: {scores['exact_match']:.2%}; CER: {scores['cer']:.2%}.")
    print(f"Kết quả: {output}")


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
