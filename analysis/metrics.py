"""Chỉ tính chỉ số; không đọc ảnh, nạp model hoặc tạo báo cáo."""

import math
import statistics
import unicodedata

from utils import OCRError


# Đưa dấu về NFC nhưng giữ nguyên hoa/thường và khoảng trắng.
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# Đếm số phép thêm, xóa, thay ký tự tối thiểu để biến đáp án thành dự đoán.
def edit_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for row, character in enumerate(reference, 1):
        current = [row]
        for column, predicted in enumerate(prediction, 1):
            current.append(min(
                previous[column] + 1, current[column - 1] + 1,
                previous[column - 1] + (character != predicted),
            ))
        previous = current
    return previous[-1]


# Chấm Exact Match và CER theo tổng ký tự, không trung bình CER từng ảnh.
def text_metrics(references: list[str], predictions: list[str]) -> dict:
    if not references or len(references) != len(predictions):
        raise OCRError("Số đáp án và dự đoán phải bằng nhau, lớn hơn 0.")
    references = [normalize_text(text) for text in references]
    predictions = [normalize_text(text) for text in predictions]
    if any(not text.strip() for text in references):
        raise OCRError("Đáp án dùng để chấm không được rỗng.")
    correct = sum(truth == predicted for truth, predicted in zip(references, predictions))
    edits = sum(edit_distance(truth, predicted) for truth, predicted in zip(references, predictions))
    characters = sum(map(len, references))
    return {
        "samples": len(references), "correct": correct,
        "exact_match": correct / len(references), "cer": edits / characters,
        "total_edits": edits, "reference_characters": characters,
    }


# Tóm tắt thời gian dự đoán từng ảnh; P95 dùng cách lấy hạng gần nhất.
def latency_metrics(milliseconds: list[float]) -> dict:
    if not milliseconds or any(not math.isfinite(value) or value < 0 for value in milliseconds):
        raise OCRError("Thời gian dự đoán phải hữu hạn, không âm và có ít nhất một mẫu.")
    return {
        "mean_ms": statistics.mean(milliseconds),
        "median_ms": statistics.median(milliseconds),
        "p95_ms": sorted(milliseconds)[math.ceil(0.95 * len(milliseconds)) - 1],
    }


# Chọn model theo Exact Match validation cao nhất, dùng CER khi bằng điểm.
def is_better(candidate: dict, best: dict | None) -> bool:
    return best is None or (
        candidate["exact_match"], -candidate["cer"]
    ) > (best["exact_match"], -best["cer"])


# Bỏ dấu chỉ để phân loại lỗi chuỗi; điểm chính vẫn giữ nguyên dấu/hoa thường.
def remove_marks(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(char for char in unicodedata.normalize("NFD", text) if not unicodedata.combining(char))


# Gán mỗi mẫu sai vào một nhóm để thống kê không bị đếm trùng.
def error_category(reference: str, prediction: str) -> str:
    if reference == prediction:
        return "Đúng"
    if not prediction:
        return "Dự đoán rỗng"
    if reference.casefold() == prediction.casefold():
        return "Chỉ khác hoa/thường"
    if remove_marks(reference) == remove_marks(prediction):
        return "Chỉ khác dấu/đ"
    if remove_marks(reference).casefold() == remove_marks(prediction).casefold():
        return "Khác dấu và hoa/thường"
    return "Khác nội dung/khoảng trắng"


# Chia theo độ dài đáp án để thấy nhóm chữ ngắn hoặc dài đang sai nhiều.
def length_group(text: str) -> str:
    if len(text) == 1:
        return "1"
    if len(text) <= 3:
        return "2–3"
    if len(text) <= 6:
        return "4–6"
    return "7+"
