"""Full fine-tuning theo epoch; chấm toàn bộ validation sau mỗi epoch."""

import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml
from tqdm import tqdm
from vietocr.tool.translate import build_model, translate

import config
from analysis.metrics import is_better, text_metrics
from finetuning.dataset import OCRDataset, make_loader
from utils import (
    OCRError, file_sha256, load_model_config, read_csv, run_cli,
    runtime_versions, save_json, write_csv,
)


# Tính tổng cross-entropy và số token thật; PAD không tham gia loss.
def batch_loss(model, batch):
    targets = batch["targets"].to(config.TRAIN_DEVICE)
    logits = model(
        batch["images"].to(config.TRAIN_DEVICE), batch["inputs"].to(config.TRAIN_DEVICE),
        batch["padding_mask"].to(config.TRAIN_DEVICE),
    )
    loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        ignore_index=0, label_smoothing=config.LABEL_SMOOTHING, reduction="sum",
    )
    if not torch.isfinite(loss):
        raise OCRError("Loss không hữu hạn; hãy kiểm tra dữ liệu hoặc giảm learning rate.")
    return loss, int(targets.ne(0).sum().item())


# Duyệt hết train một lần và cập nhật toàn bộ tham số trong từng batch.
def train_epoch(model, loader, optimizer):
    model.train()
    loss_sum, tokens, samples = 0.0, 0, 0
    for batch in tqdm(loader, desc="Train"):
        optimizer.zero_grad(set_to_none=True)
        loss, count = batch_loss(model, batch)
        (loss / count).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.detach().item()
        tokens += count
        samples += len(batch["references"])
    if samples != len(loader.dataset):
        raise OCRError("Epoch chưa duyệt đủ train; không tiếp tục với số liệu này.")
    return loss_sum / tokens


# Chấm hết validation bằng greedy, tính loss theo token và Exact Match/CER theo chuỗi.
@torch.inference_mode()
def validate(model, vocab, loader):
    model.eval()
    references, predictions = [], []
    loss_sum, tokens = 0.0, 0
    for batch in tqdm(loader, desc="Validation"):
        loss, count = batch_loss(model, batch)
        loss_sum += loss.item()
        tokens += count
        encoded, _ = translate(batch["images"].to(config.TRAIN_DEVICE), model, config.MAX_TEXT_LENGTH)
        references.extend(batch["references"])
        predictions.extend(vocab.batch_decode(encoded.tolist()))
    if len(references) != len(loader.dataset):
        raise OCRError("Validation chưa được chấm đủ mẫu.")
    return {**text_metrics(references, predictions), "val_loss": loss_sum / tokens}


# Lưu checkpoint qua tệp tạm để giữ bản cũ nếu quá trình ghi bị ngắt.
def save_weights(model, path):
    temporary = path.with_suffix(".tmp")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    temporary.replace(path)


# Khởi tạo đúng pretrained, train theo EPOCHS và chọn best bằng validation, không dùng test.
def main():
    if min(config.EPOCHS, config.BATCH_SIZE) < 1 or not 0 < config.LEARNING_RATE < 1:
        raise OCRError("EPOCHS/BATCH_SIZE phải dương và LEARNING_RATE phải nằm trong (0, 1).")
    torch.set_num_threads(config.CPU_THREADS)
    if config.TRAIN_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise OCRError("PyTorch hiện không dùng được CUDA. Hãy train trên máy/Colab có GPU phù hợp.")
    if config.CHECKPOINTS.exists() and any(config.CHECKPOINTS.iterdir()):
        raise OCRError("Thư mục checkpoint đã có dữ liệu; không ghi đè lần train cũ.")
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model_config = load_model_config(config.BEFORE / "model.yml", config.TRAIN_DEVICE)
    baseline = json.loads((config.BEFORE / "metrics.json").read_text(encoding="utf-8"))
    initial_hash = file_sha256(Path(model_config["weights"]))
    if initial_hash != baseline["weights_sha256"]:
        raise OCRError("Trọng số pretrained khác model đã benchmark.")
    train = OCRDataset(config.CROPS / "train", model_config)
    val = OCRDataset(config.CROPS / "val", model_config)
    train_sources = {row["source_image"] for row in train.samples}
    val_sources = {row["source_image"] for row in val.samples}
    test_sources = {row["source_image"] for row in read_csv(config.CROPS / "test/samples.csv")}
    if train_sources & val_sources or (train_sources | val_sources) & test_sources:
        raise OCRError("Ảnh nguồn bị trùng giữa train, validation hoặc test.")
    model, vocab = build_model(model_config)
    model.load_state_dict(torch.load(model_config["weights"], map_location="cpu", weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    val_loader = make_loader(val, vocab)
    config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    exported = copy.deepcopy(model_config)
    exported["weights"] = "best.pth"
    (config.CHECKPOINTS / "model.yml").write_text(yaml.safe_dump(exported, allow_unicode=True), encoding="utf-8")
    save_json(config.CHECKPOINTS / "run.json", {
        "epochs": config.EPOCHS, "batch_size": config.BATCH_SIZE, "learning_rate": config.LEARNING_RATE,
        "device": config.TRAIN_DEVICE, "cpu_threads": torch.get_num_threads(),
        "seed": config.SEED, "initial_weights_sha256": initial_hash,
        "train_samples": len(train), "val_samples": len(val),
        "weight_decay": config.WEIGHT_DECAY, "label_smoothing": config.LABEL_SMOOTHING,
        "gradient_clip": config.GRADIENT_CLIP, "environment": runtime_versions(),
        "train_samples_sha256": file_sha256(config.CROPS / "train/samples.csv"),
        "val_samples_sha256": file_sha256(config.CROPS / "val/samples.csv"),
    })
    history, best = [], None
    # Epoch 0 là mốc validation của pretrained; epoch 1..EPOCHS mới cập nhật model.
    for epoch in range(config.EPOCHS + 1):
        train_loss = train_epoch(model, make_loader(train, vocab, shuffle=True), optimizer) if epoch else None
        scores = validate(model, vocab, val_loader)
        if is_better(scores, best):
            best = {"epoch": epoch, **scores}
            save_weights(model, config.CHECKPOINTS / "best.pth")
        save_weights(model, config.CHECKPOINTS / "last.pth")
        history.append({"epoch": epoch, "train_loss": train_loss, **scores})
        write_csv(config.CHECKPOINTS / "history.csv", list(history[0]), history)
        save_json(config.CHECKPOINTS / "best_metrics.json", best)
        loss_label = "—" if train_loss is None else f"{train_loss:.4f}"
        print(f"Epoch {epoch}/{config.EPOCHS}: train loss={loss_label}, val loss={scores['val_loss']:.4f}, "
              f"Exact Match={scores['exact_match']:.2%}, CER={scores['cer']:.2%}.")
    save_json(config.CHECKPOINTS / "summary.json", {"epochs_completed": config.EPOCHS, "best": best})
    print(f"Đã train xong. Checkpoint tốt nhất ở epoch {best['epoch']}: {config.CHECKPOINTS / 'best.pth'}")


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
