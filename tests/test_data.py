"""Kiểm tra crop, nhãn, cách tạo batch; không tải pretrained hoặc train VietOCR."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from vietocr.model.vocab import Vocab

import config
from finetuning.dataset import OCRDataset, collate_batch, make_loader
from preprocessing.data import prepare_split, read_regions
from utils import read_csv, write_csv

BOX = "0,0,20,0,20,10,0,10,"


class DataTests(unittest.TestCase):
    # Cắt đúng kích thước, giữ nhãn ngoài vocab trên test và ghi đủ vùng bị loại.
    def test_crop_and_test_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unseen_test_images").mkdir()
            (root / "labels").mkdir()
            Image.new("RGB", (32, 16), "white").save(root / "unseen_test_images/im1501.jpg")
            (root / "labels/gt_1501.txt").write_text(
                BOX + "  A\u0300,20°  \n" + BOX + "###\n" + "0,0,0,0,0,0,0,0,A\n", encoding="utf-8",
            )
            output = root / "prepared"
            summary = prepare_split(root, output, "test")
            self.assertEqual((summary["accepted"], summary["excluded"]), (1, 2))
            rows = read_csv(output / "samples.csv")
            self.assertEqual(rows[0]["ground_truth"], "  À,20°  ")
            with Image.open(output / rows[0]["crop"]) as image:
                self.assertEqual(image.size, (20, 10))
            with self.assertRaisesRegex(ValueError, "Đã có dữ liệu"):
                prepare_split(root, output, "test")

    # Tọa độ hỏng phải dừng và chỉ đúng dòng; không âm thầm loại để nâng điểm.
    def test_invalid_annotation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt_1.txt"
            path.write_text("nan,0,20,0,20,10,0,10,A", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dòng 1"):
                list(read_regions(path))

    # Nhãn teacher forcing phải đúng SOS/EOS/PAD và thứ tự chiều batch/thời gian.
    def test_shifted_targets_and_padding(self):
        image = torch.zeros(3, 32, 40)
        batch = collate_batch([(image, "AB"), (image, "A")], Vocab("AB"))
        self.assertEqual(batch["inputs"].tolist(), [[1, 1], [4, 4], [5, 0]])
        self.assertEqual(batch["targets"].tolist(), [[4, 5, 2], [4, 2, 0]])
        self.assertEqual(batch["padding_mask"].tolist(), [[False, False, False], [False, False, True]])

    # Mỗi epoch giữ đủ mẫu, kể cả batch cuối và nhóm chiều rộng chỉ có một ảnh.
    def test_loader_keeps_every_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, width in enumerate([8, 8, 8, 20]):
                name = f"{index}.png"
                Image.new("RGB", (width, 16), "white").save(root / name)
                rows.append({"crop": name, "ground_truth": "A", "source_image": "im0001.jpg"})
            write_csv(root / "samples.csv", list(rows[0]), rows)
            settings = {"vocab": "A", "dataset": {
                "image_height": 32, "image_min_width": 32, "image_max_width": 512,
            }}
            dataset = OCRDataset(root, settings)
            with patch.object(config, "BATCH_SIZE", 2):
                for shuffle in (False, True):
                    loader = make_loader(dataset, Vocab("A"), shuffle)
                    indices = [index for batch in loader.batch_sampler for index in batch]
                    self.assertEqual(sorted(indices), [0, 1, 2, 3])
                    self.assertEqual(sum(len(batch["references"]) for batch in loader), 4)
            self.assertEqual(dataset[0][0].dtype, torch.float32)
            self.assertTrue(np.isfinite(dataset[0][0].numpy()).all())


if __name__ == "__main__":
    unittest.main()
