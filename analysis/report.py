"""Đọc kết quả benchmark, xuất bảng lỗi và so sánh trước/sau; không chạy model."""

import os
from collections import Counter, defaultdict
from pathlib import Path

import config
from analysis.metrics import edit_distance, error_category, latency_metrics, length_group, normalize_text, text_metrics
from utils import OCRError, VietnameseParser, read_csv, run_cli, save_json, write_csv


# Đọc dự đoán, kiểm tra định danh/hash và chấm lại từ văn bản gốc.
def read_results(path):
    required = ["crop", "source_image", "line", "ground_truth", "prediction", "crop_sha256", "predict_ms"]
    rows = read_csv(path, required)
    seen = set()
    for row in rows:
        identity = (row["source_image"], row["line"])
        digest = row["crop_sha256"]
        if not all(identity) or identity in seen or len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise OCRError(f"Kết quả có định danh trùng/thiếu hoặc hash ảnh sai: {path}.")
        seen.add(identity)
        row["ground_truth"] = normalize_text(row["ground_truth"])
        row["prediction"] = normalize_text(row["prediction"])
        row["edit_distance"] = edit_distance(row["ground_truth"], row["prediction"])
        row["correct"] = row["ground_truth"] == row["prediction"]
        row["predict_ms"] = float(row["predict_ms"])
        row["error_type"] = error_category(row["ground_truth"], row["prediction"])
    return rows


# Gọi cùng công thức chấm được dùng trong benchmark và validation.
def score_rows(rows):
    return text_metrics([row["ground_truth"] for row in rows], [row["prediction"] for row in rows])


# Ghép đúng cùng crop/nhãn rồi ghi dự đoán trước/sau và các mẫu cải thiện/suy giảm.
def compare_results(before, after):
    old = {(row["source_image"], row["line"]): row for row in before}
    new = {(row["source_image"], row["line"]): row for row in after}
    if len(old) != len(before) or len(new) != len(after) or old.keys() != new.keys():
        raise OCRError("Hai kết quả phải có đúng cùng tập mẫu và không trùng định danh.")
    changes = []
    for key, row in new.items():
        previous = old[key]
        if (row["ground_truth"], row["crop_sha256"]) != (previous["ground_truth"], previous["crop_sha256"]):
            raise OCRError(f"Ảnh hoặc đáp án khác nhau giữa hai benchmark: {key}.")
        status = "Bằng số lỗi"
        if row["correct"] and not previous["correct"]:
            status = "Sai thành đúng"
        elif previous["correct"] and not row["correct"]:
            status = "Đúng thành sai"
        elif row["edit_distance"] < previous["edit_distance"]:
            status = "Ít lỗi hơn"
        elif row["edit_distance"] > previous["edit_distance"]:
            status = "Nhiều lỗi hơn"
        changes.append({**row, "baseline_prediction": previous["prediction"],
                        "baseline_edit_distance": previous["edit_distance"], "status": status})
    previous, current = score_rows(before), score_rows(after)
    counts = Counter(row["status"] for row in changes)
    return changes, {
        "exact_match_delta": current["exact_match"] - previous["exact_match"],
        "cer_delta": current["cer"] - previous["cer"], "changes": dict(counts),
    }


# Đổi đường dẫn crop từ thư mục benchmark sang thư mục báo cáo tách riêng.
def rebase_paths(rows, source, output):
    return [{**row, "crop": Path(os.path.relpath((source / row["crop"]).resolve(), output)).as_posix()}
            for row in rows]


# Xuất báo cáo ngắn, các nhóm điểm và bảng lỗi để người chạy tự đối chiếu ảnh.
def main():
    parser = VietnameseParser(description=__doc__)
    parser.add_argument("phase", choices=["before", "after"])
    phase = parser.parse_args().phase
    source = config.BEFORE if phase == "before" else config.AFTER
    output = config.REPORTS / phase
    rows = read_results(source / "predictions.csv")
    text_scores = score_rows(rows)
    scores = {**text_scores, **latency_metrics([row["predict_ms"] for row in rows])}
    errors = sorted((row for row in rows if not row["correct"]), key=lambda row: row["edit_distance"], reverse=True)
    scores["error_types"] = dict(Counter(row["error_type"] for row in errors))
    fields = ["crop", "source_image", "line", "ground_truth", "prediction", "edit_distance", "error_type"]
    write_csv(output / "errors.csv", fields, rebase_paths(errors, source, output))
    groups = defaultdict(list)
    for row in rows:
        groups[length_group(row["ground_truth"])].append(row)
    grouped = [{"length": label, **score_rows(group)} for label, group in groups.items()]
    write_csv(output / "by_length.csv", ["length", *text_scores], grouped)
    report = [f"# Benchmark {phase}", f"Số mẫu: {scores['samples']}; số mẫu sai: {len(errors)}.",
              f"Exact Match: {scores['exact_match']:.2%}; CER: {scores['cer']:.2%}.",
              f"Mean / median / P95: {scores['mean_ms']:.2f} / {scores['median_ms']:.2f} / {scores['p95_ms']:.2f} ms.",
              f"Các nhóm lỗi: {scores['error_types']}."]
    if phase == "after":
        changes, comparison = compare_results(read_results(config.BEFORE / "predictions.csv"), rows)
        scores["comparison"] = comparison
        write_csv(output / "paired_changes.csv", fields + ["baseline_prediction", "baseline_edit_distance", "status"],
                  rebase_paths(changes, source, output))
        report += [f"Exact Match thay đổi: {100 * comparison['exact_match_delta']:+.2f} điểm phần trăm.",
                   f"CER thay đổi: {100 * comparison['cer_delta']:+.2f} điểm phần trăm.",
                   f"Thay đổi từng mẫu: {comparison['changes']}."]
    report.append(
        "Exact Match cao hơn/CER thấp hơn là tốt hơn. Mở crop sai trước khi kết luận nguyên nhân; "
        "chỉ so tốc độ khi cùng điều kiện đo."
    )
    save_json(output / "metrics.json", scores)
    (output / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print(f"Báo cáo: {output / 'report.md'}")


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
