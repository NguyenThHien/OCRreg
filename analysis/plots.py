"""Vẽ đường học và so sánh test từ kết quả đã lưu; không chạy lại model."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from analysis.report import compare_results, read_results, score_rows
from utils import OCRError, VietnameseParser, read_csv, run_cli


# Kiểm tra đầu vào để báo rõ bước cần chạy khi chưa có dữ liệu vẽ.
def require_files(paths, instruction):
    for path in paths:
        if not path.is_file():
            raise OCRError(f"Chưa có tệp {path}. {instruction}")


# Lưu cùng biểu đồ dưới dạng PNG để xem và SVG để đưa vào báo cáo.
def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for extension in ("png", "svg"):
            target = path.with_suffix(f".{extension}")
            figure.savefig(target, dpi=180, bbox_inches="tight")
            print(f"Biểu đồ: {target}")
    finally:
        plt.close(figure)


# Vẽ loss và điểm validation theo epoch; epoch 0 chỉ đo pretrained, không có train loss.
def plot_training():
    history = config.CHECKPOINTS / "history.csv"
    best_path = config.CHECKPOINTS / "best_metrics.json"
    require_files([history, best_path], "Hãy chạy finetuning.train trước.")
    rows = read_csv(history, ["epoch", "train_loss", "val_loss", "exact_match", "cer"])
    if not rows:
        raise OCRError("Lịch sử huấn luyện chưa có epoch để vẽ.")
    epochs = [int(row["epoch"]) for row in rows]
    best_epoch = json.loads(best_path.read_text(encoding="utf-8"))["epoch"]
    if best_epoch not in epochs:
        raise OCRError("Epoch tốt nhất không có trong lịch sử huấn luyện.")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].plot(epochs, [float(row["train_loss"] or "nan") for row in rows], "o-", label="Train")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], "o-", label="Validation")
    axes[0].set(title="Loss trung bình mỗi token", ylabel="Cross-entropy")
    axes[0].legend()
    for axis, metric, title in zip(axes[1:], ("exact_match", "cer"),
                                   ("Validation: Exact Match ↑", "Validation: CER ↓")):
        axis.plot(epochs, [100 * float(row[metric]) for row in rows], "o-")
        axis.set(title=title, ylabel="%")
        axis.set_ylim(bottom=0)
    for axis in axes:
        axis.axvline(best_epoch, color="tab:green", linestyle="--", alpha=0.7)
        axis.set(xlabel="Epoch", xticks=epochs)
        axis.grid(alpha=0.2)
    figure.suptitle(f"Đường học — nét đứt: checkpoint tốt nhất ở epoch {best_epoch}; epoch 0 chưa train")
    save_figure(figure, config.REPORTS / "training/learning_curves")


# Chấm lại hai CSV trên cùng crop/nhãn/hash rồi vẽ điểm test và số mẫu được sửa/làm sai thêm.
def plot_test():
    paths = [folder / "predictions.csv" for folder in (config.BEFORE, config.AFTER)]
    require_files(paths, "Hãy hoàn tất benchmark.run before và benchmark.run after trước.")
    before, after = [read_results(path) for path in paths]
    _, comparison = compare_results(before, after)
    scores = [score_rows(before), score_rows(after)]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for axis, metric, title in zip(axes[:2], ("exact_match", "cer"),
                                   ("Test: Exact Match ↑", "Test: CER ↓")):
        values = [100 * score[metric] for score in scores]
        bars = axis.bar(["Trước", "Sau"], values, color=["tab:gray", "tab:blue"])
        axis.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=4)
        axis.set(title=title, ylabel="%", ylim=(0, max(1, max(values) * 1.2)))
        delta = 100 * comparison[f"{metric}_delta"]
        axis.set_xlabel(f"Sau − trước: {delta:+.2f} điểm phần trăm")
    labels = ["Sai thành đúng", "Đúng thành sai"]
    counts = [comparison["changes"].get(label, 0) for label in labels]
    bars = axes[2].bar(labels, counts, color=["tab:green", "tab:red"])
    axes[2].bar_label(bars, padding=4)
    axes[2].set(title="Thay đổi Exact Match từng crop", ylabel="Số crop",
                ylim=(0, max(1, max(counts) * 1.2)))
    figure.suptitle(f"Cùng {scores[0]['samples']:,} crop test — pretrained và best.pth")
    save_figure(figure, config.REPORTS / "after/test_comparison")


# Chọn biểu đồ cần tạo; mọi đầu vào đều lấy từ tệp do người dùng đã chạy tạo ra.
def main():
    parser = VietnameseParser(description=__doc__)
    parser.add_argument("stage", choices=["training", "test"])
    stage = parser.parse_args().stage
    if stage == "training":
        plot_training()
    else:
        plot_test()


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
