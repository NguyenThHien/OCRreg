"""Đọc crop đã chuẩn bị và tạo batch cùng chiều rộng cho train/validation."""

import random
from collections import defaultdict
from functools import partial

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from vietocr.tool.translate import process_image, resize

import config
from analysis.metrics import normalize_text
from utils import OCRError, read_csv


class OCRDataset(Dataset):
    # Đọc nhãn, kiểm tra đường dẫn/vocab và lưu chiều rộng sau resize để gom batch.
    def __init__(self, root, model_config):
        self.size = model_config["dataset"]
        self.samples = read_csv(root / "samples.csv", ["crop", "ground_truth", "source_image"])
        if not self.samples:
            raise OCRError(f"Không có mẫu dữ liệu: {root}.")
        seen = set()
        vocabulary = set(model_config["vocab"])
        for row in self.samples:
            row["text"] = normalize_text(row["ground_truth"])
            row["path"] = (root / row["crop"]).resolve()
            if not row["path"].is_relative_to(root.resolve()) or row["path"] in seen:
                raise OCRError(f"Đường dẫn crop nằm ngoài tập hoặc bị lặp: {row['crop']}.")
            seen.add(row["path"])
            if not row["text"].strip() or set(row["text"]) - vocabulary:
                raise OCRError(f"Nhãn rỗng hoặc ngoài vocab: {row['crop']}.")
            if len(row["text"]) > config.MAX_TEXT_LENGTH:
                raise OCRError(f"Nhãn dài hơn giới hạn giải mã: {row['crop']}.")
            with Image.open(row["path"]) as image:
                row["width"], _ = resize(image.width, image.height, *self.dimensions())

    # Lấy ba tham số resize theo thứ tự mà VietOCR yêu cầu.
    def dimensions(self):
        return self.size["image_height"], self.size["image_min_width"], self.size["image_max_width"]

    # Trả tổng số crop để kiểm tra train/validation đã duyệt đủ mỗi epoch.
    def __len__(self):
        return len(self.samples)

    # Đọc ảnh RGB, resize đúng VietOCR và chuyển về tensor float trong khoảng 0–1.
    def __getitem__(self, index):
        row = self.samples[index]
        with Image.open(row["path"]) as image:
            pixels = process_image(image.convert("RGB"), *self.dimensions())
        return torch.from_numpy(pixels).float(), row["text"]


# Tạo nhãn teacher forcing: đầu vào SOS+chữ, đầu ra chữ+EOS; bỏ PAD khi tính loss.
def collate_batch(batch, vocab):
    images, texts = zip(*batch)
    tokens = [vocab.encode(text) for text in texts]
    length = max(map(len, tokens)) - 1
    inputs = torch.full((len(batch), length), vocab.pad, dtype=torch.long)
    targets = torch.full_like(inputs, vocab.pad)
    for row, encoded in enumerate(tokens):
        inputs[row, :len(encoded) - 1] = torch.tensor(encoded[:-1])
        targets[row, :len(encoded) - 1] = torch.tensor(encoded[1:])
    return {
        "images": torch.stack(images), "inputs": inputs.T,
        "targets": targets, "padding_mask": inputs.eq(vocab.pad), "references": list(texts),
    }


# Gom cùng chiều rộng, xáo train mỗi lần gọi và giữ cả batch thiếu ở cuối nhóm.
def make_loader(dataset, vocab, shuffle=False):
    groups = defaultdict(list)
    for index, row in enumerate(dataset.samples):
        groups[row["width"]].append(index)
    batches = []
    for indices in groups.values():
        if shuffle:
            random.shuffle(indices)
        batches.extend(indices[start:start + config.BATCH_SIZE] for start in range(0, len(indices), config.BATCH_SIZE))
    if shuffle:
        random.shuffle(batches)
    return DataLoader(dataset, batch_sampler=batches, collate_fn=partial(collate_batch, vocab=vocab))
