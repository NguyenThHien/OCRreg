"""Kiểm tra công thức chỉ số và tính hợp lệ khi so sánh hai benchmark."""

import unittest

from analysis.metrics import edit_distance, error_category, is_better, latency_metrics, text_metrics
from analysis.report import compare_results


# Tạo kết quả tối thiểu giống một dòng benchmark, không chạy model.
def result(line, truth, prediction):
    return {
        "source_image": "im1501.jpg", "line": str(line), "crop_sha256": "0" * 64,
        "ground_truth": truth, "prediction": prediction, "correct": truth == prediction,
        "edit_distance": edit_distance(truth, prediction),
    }


class MetricsTests(unittest.TestCase):
    # CER tính theo tổng ký tự và vẫn đếm dự đoán thừa; NFC không bỏ dấu/hoa.
    def test_strict_scores_and_corpus_cer(self):
        scores = text_metrics(["A", "BBBBBBBBB", "Á"], ["AA", "BBBBBBBBB", "A\u0301"])
        self.assertEqual(scores["correct"], 2)
        self.assertAlmostEqual(scores["cer"], 1 / 11)
        self.assertGreater(text_metrics(["A"], ["AAAA"])["cer"], 1)
        self.assertEqual(text_metrics(["A"], ["a"])["correct"], 0)
        self.assertEqual(text_metrics([" A "], ["A"])["correct"], 0)

    # Median/P95 và các giá trị thời gian sai phải được xử lý rõ ràng.
    def test_latency_and_invalid_measurements(self):
        scores = latency_metrics(list(range(1, 21)))
        self.assertEqual(scores["median_ms"], 10.5)
        self.assertEqual(scores["p95_ms"], 19)
        for values in ([], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                latency_metrics(values)

    # Mỗi mẫu sai thuộc đúng một nhóm khác biệt chuỗi.
    def test_error_categories(self):
        self.assertEqual(error_category("ĐỎ", "DO"), "Chỉ khác dấu/đ")
        self.assertEqual(error_category("NHÀ", "nha"), "Khác dấu và hoa/thường")
        self.assertEqual(error_category("ABC", ""), "Dự đoán rỗng")

    # Ghép theo định danh thay vì thứ tự dòng, đếm cả cải thiện và suy giảm.
    def test_paired_changes(self):
        before = [result(1, "A", "B"), result(2, "CCC", "CCC")]
        after = [result(2, "CCC", "CC"), result(1, "A", "A")]
        changes, scores = compare_results(before, after)
        self.assertEqual(scores["exact_match_delta"], 0)
        self.assertEqual(scores["changes"], {"Đúng thành sai": 1, "Sai thành đúng": 1})
        self.assertEqual(len(changes), 2)
        for invalid in (
            after[:1], after + after[:1],
            [{**after[0], "crop_sha256": "1" * 64}, after[1]],
            [{**after[0], "ground_truth": "XYZ"}, after[1]],
        ):
            with self.assertRaises(ValueError):
                compare_results(before, invalid)

    # Chọn Exact Match trước CER và không thay checkpoint khi bằng cả hai điểm.
    def test_checkpoint_selection(self):
        best = {"exact_match": 0.7, "cer": 0.2}
        self.assertTrue(is_better(best, None))
        self.assertTrue(is_better({"exact_match": 0.8, "cer": 0.3}, best))
        self.assertTrue(is_better({"exact_match": 0.7, "cer": 0.1}, best))
        self.assertFalse(is_better(best, best))


if __name__ == "__main__":
    unittest.main()
