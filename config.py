"""Chỉnh tham số tại đây; các script không khai báo lại kích thước hay lịch train."""

from pathlib import Path

# Nguồn và nơi lưu kết quả.
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "vintext_original/vietnamese"
SOURCE_FOLDERS = {"train": "train_images", "val": "test_image", "test": "unseen_test_images"}
SOURCE_RANGES = {"train": (1, 1200), "val": (1201, 1500), "test": (1501, 2000)}
OUTPUT = ROOT / "output"
CROPS = OUTPUT / "crops"
BEFORE = OUTPUT / "benchmark/before"
AFTER = OUTPUT / "benchmark/after"
CHECKPOINTS = OUTPUT / "finetuning"
REPORTS = OUTPUT / "analysis"

# Model và kích thước đầu vào sau resize, đơn vị pixel.
MODEL_NAME = "vgg_seq2seq"
IMAGE_HEIGHT = 32
IMAGE_MIN_WIDTH = 32
IMAGE_MAX_WIDTH = 512
MAX_TEXT_LENGTH = 128

# Benchmark luôn dùng toàn bộ test, batch 1.
BENCHMARK_DEVICE = "cpu"
CPU_THREADS = 4  # Dùng chung cho benchmark và train trên CPU.
WARMUP_RUNS = 3

# Full fine-tuning: duyệt hết train, rồi chấm hết validation sau mỗi epoch.
TRAIN_DEVICE = "cpu"
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP = 1.0
SEED = 42
