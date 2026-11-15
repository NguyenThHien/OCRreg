# VietOCR trên VinText

**Đã hoàn tất full fine-tuning 10 epoch trên CPU; checkpoint tốt nhất ở epoch 9.** Số liệu bên dưới đã đối chiếu với các tệp bạn chạy tạo ra. Bước tiếp theo là benchmark `best.pth` trên đúng bộ test cũ; chưa có kết quả test sau fine-tuning.

## Đọc code theo thứ tự này

| File | Trách nhiệm |
| --- | --- |
| `config.py` | Toàn bộ đường dẫn, kích thước ảnh, batch size, epoch, learning rate và thiết bị. Chỉnh tham số tại đây. |
| `preprocessing/data.py` | Đọc nhãn/tứ giác, cắt crop, loại vùng không hợp lệ, tạo dữ liệu test hoặc train/val. |
| `benchmark/run.py` | Nạp model và dự đoán **toàn bộ test**. `before` dùng pretrained; `after` dùng model tốt nhất. |
| `analysis/metrics.py` | Các công thức: Levenshtein, Exact Match, CER, mean/median/P95, phân nhóm lỗi và chọn checkpoint. |
| `analysis/report.py` | Đọc CSV, gọi các hàm chỉ số, xuất bảng lỗi và so sánh trước/sau. |
| `analysis/plots.py` | Vẽ đường học trên validation và biểu đồ test trước/sau từ kết quả đã lưu; không chạy model. |
| `finetuning/dataset.py` | Đọc crop/CSV, resize, tạo nhãn SOS/EOS/PAD và batch cùng chiều rộng. |
| `finetuning/train.py` | Full fine-tuning: duyệt hết train rồi chấm hết validation sau mỗi epoch. |
| `utils.py` | Đọc/ghi CSV/JSON/YAML, SHA-256 và thông báo lỗi tiếng Việt. |
| `tests/` | Kiểm tra công thức, crop và batch bằng dữ liệu nhỏ tự tạo. |

`__init__.py` chỉ đánh dấu package. Dữ liệu gốc và `.venv` được giữ nguyên. Các script minh họa một ảnh, code cũ trùng chức năng và `outputs` cũ đã được xóa theo yêu cầu.

```text
output/
  crops/test/          ảnh và đáp án test cố định
  crops/train/         ảnh và đáp án train
  crops/val/           ảnh và đáp án validation
  benchmark/before/    dự đoán/chỉ số/trọng số pretrained
  benchmark/after/     dự đoán/chỉ số/trọng số sau fine-tuning
  analysis/before/     báo cáo trước
  analysis/after/      báo cáo sau và bảng so sánh
  analysis/training/   biểu đồ loss và điểm validation theo epoch
  finetuning/          checkpoint, lịch sử epoch
  logs/                chi tiết kỹ thuật khi có lỗi
```

## Tham số chính trong config.py

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `IMAGE_HEIGHT` | 32 px | Chiều cao sau resize. |
| `IMAGE_MIN_WIDTH` / `IMAGE_MAX_WIDTH` | 32 / 512 px | Giới hạn chiều rộng theo cách resize của VietOCR. |
| `MAX_TEXT_LENGTH` | 128 | Giới hạn giải mã greedy. |
| `BATCH_SIZE` | 4 | Giảm nhu cầu RAM khi train CPU; ảnh cùng chiều rộng, giữ batch thiếu. |
| `EPOCHS` | 10 | 10 lần duyệt hết train; validation sau mỗi lần. |
| `LEARNING_RATE` | 0.0001 | Learning rate cố định của AdamW. |
| `WEIGHT_DECAY` | 0.01 | Điều chuẩn trọng số. |
| `LABEL_SMOOTHING` | 0.1 | Làm mềm nhãn khi tính cross-entropy. |
| `GRADIENT_CLIP` | 1.0 | Giới hạn chuẩn gradient. |
| `BENCHMARK_DEVICE` / `TRAIN_DEVICE` | cpu / cpu | Theo lựa chọn chạy CPU trên máy hiện tại. |
| `CPU_THREADS` / `WARMUP_RUNS` | 4 / 3 | Thread CPU dùng chung benchmark/train; số lượt khởi động chỉ áp dụng benchmark. |
| `SEED` | 42 | Nguồn ngẫu nhiên cho model/thứ tự batch. |

Đây là cấu hình khởi đầu, chưa chứng minh tối ưu. Kích thước model thực tế được lưu trong `model.yml`; giữ nguyên cấu hình ảnh và dữ liệu trong thí nghiệm trước/sau. Chọn hyperparameter bằng validation, không thử nhiều cấu hình rồi chọn theo test.

## Dữ liệu và cách đo

Nguồn: `vintext_original/vietnamese`. `train_images` (mã 1–1200) là train; `test_image` (1201–1500) được dùng làm validation; `unseen_test_images` (1501–2000) là test giữ riêng. `vintext` cũ không dùng trong quy trình này.

Crop theo tứ giác gốc, chỉnh phối cảnh, không tự thêm padding/xoay. Chuẩn hóa NFC nhưng giữ hoa/thường, dấu và khoảng trắng. Loại nhãn `###`, rỗng hoặc tứ giác không hợp lệ và ghi lý do. Train/val loại thêm nhãn ngoài vocab hoặc quá giới hạn giải mã; test giữ cả các nhãn đó. Nhãn sai định dạng làm script dừng để sửa. Báo số mẫu giữ/loại trong `preparation.json`.

| Chỉ số | Đo thế nào? | Vì sao cần? |
| --- | --- | --- |
| Exact Match | Số chuỗi đúng hoàn toàn / số crop | Tỷ lệ dùng được nguyên chuỗi; cao hơn tốt hơn. |
| CER | Tổng phép thêm/xóa/thay / tổng ký tự đáp án | Mức sai ký tự; thấp hơn tốt hơn. Không trung bình CER từng crop. |
| Mean / median / P95 latency | Trung bình / trung vị / hạng `ceil(0.95 × N)` của thời gian ms/crop | Chi phí trung bình, trường hợp điển hình và phần đuôi chậm. |
| Train / val loss | Tổng cross-entropy / tổng token khác PAD, gồm EOS | Theo dõi học/overfitting; không thay điểm chuỗi giải mã. |
| Sai thành đúng / đúng thành sai | Ghép cùng định danh, nhãn và SHA-256 ảnh | Đếm mẫu được sửa và mẫu bị làm sai thêm. |

Chỉ bấm giờ quanh tiền xử lý ảnh → suy luận → giải mã. Không tính đọc ảnh, SHA-256, tải model, cắt ảnh hoặc warmup. Benchmark batch 1, greedy; CUDA được đồng bộ khi đo. So tốc độ phải cùng máy, thiết bị, thread và môi trường; thông tin lưu trong `metrics.json`.

CER có thể trên 100% khi dự đoán thừa nhiều. Exact Match từ 61% lên 65% là +4 **điểm phần trăm**; đây là ví dụ, không phải kết quả đo. Chỉ khác dấu trong chuỗi chưa chứng minh ảnh bị mờ: cần mở crop kiểm tra. Đây là nhận dạng trên crop từ nhãn, chưa đánh giá phát hiện chữ/OCR đầu cuối. Chưa xác minh pretrained có từng gặp test này trước đó hay không.

## Tự chạy từng bước

Mở PowerShell tại `D:\OCRreg`. Các lệnh dùng cùng thư mục dự án:

```powershell
# 0. Kiểm tra code, không tải/train model VietOCR.
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v

# 1. Tạo test một lần, giữ nguyên đến hết thí nghiệm.
.\.venv\Scripts\python.exe -m preprocessing.data test

# 2. Benchmark toàn bộ test bằng pretrained.
.\.venv\Scripts\python.exe -m benchmark.run before

# 3. Phân tích kết quả trước.
.\.venv\Scripts\python.exe -m analysis.report before

# 4. Chuẩn bị train và validation.
.\.venv\Scripts\python.exe -m preprocessing.data train

# 5. Train theo epoch, validation sau mỗi epoch.
.\.venv\Scripts\python.exe -m finetuning.train

# 6. Benchmark lại chính test ở bước 1 bằng best.pth.
.\.venv\Scripts\python.exe -m benchmark.run after

# 7. Phân tích và so sánh từng crop trước/sau.
.\.venv\Scripts\python.exe -m analysis.report after

# 8. Vẽ đường học (có thể chạy ngay sau bước 5).
.\.venv\Scripts\python.exe -m analysis.plots training

# 9. Vẽ so sánh test sau khi hoàn tất bước 6 và 7.
.\.venv\Scripts\python.exe -m analysis.plots test
```

Chỉ đi tiếp sau khi bước trước thành công. Theo lựa chọn của bạn, bước 5 chạy trên CPU máy hiện tại: 4 thread, batch tối đa 4, 10 epoch. CPU sẽ chậm hơn GPU; mỗi epoch vẫn duyệt đủ train và validation. Không cần cài lại PyTorch hoặc chuyển sang CUDA.

Benchmark lưu `predictions.csv`, `metrics.json`, `model.yml`, `weights.pth`. Trọng số được giữ lại để train khởi tạo đúng model đã đo; code kiểm tra SHA-256 trước khi train. Không ghi đè benchmark/crop/checkpoint đã có.

Train ghi `history.csv` theo **epoch**, `best_metrics.json`, `best.pth`, `last.pth`, `model.yml`, `run.json`; `summary.json` xuất khi hoàn tất. Epoch 0 chỉ đo validation của pretrained. Chọn best bằng Exact Match validation cao nhất, CER thấp hơn khi bằng điểm; nếu không cải thiện thì best vẫn là epoch 0. PTH chỉ chứa trọng số, chưa hỗ trợ tiếp tục chính xác optimizer của phiên bị ngắt.

Phân tích xuất `report.md`, `errors.csv`, `by_length.csv`, `metrics.json`; giai đoạn `after` có thêm `paired_changes.csv`. Đường dẫn crop trong CSV tính từ thư mục chứa CSV. Nếu Exact Match và CER thay đổi trái chiều, ghi rõ đánh đổi; kết luận chỉ áp dụng trên bộ test này.

Biểu đồ dùng Matplotlib, đã có trong môi trường hiện tại và được khai báo trong `requirements.txt`. Lệnh `training` tạo `output/analysis/training/learning_curves.png` và `.svg`: train/val loss, Exact Match validation và CER validation. Nét đứt đánh dấu checkpoint được chọn bằng Exact Match/CER, không phải bằng loss. Lệnh `test` tạo `output/analysis/after/test_comparison.png` và `.svg`: điểm test trước/sau và số crop sai thành đúng/đúng thành sai. Code kiểm tra hai kết quả có cùng định danh, đáp án và hash ảnh trước khi so sánh; không so điểm validation với điểm test. Mã vẽ mới được bổ sung, chưa chạy tạo biểu đồ.

## Nhật ký và kết quả — chỉ cập nhật sau khi bạn chạy

| Mốc | Trạng thái |
| --- | --- |
| Test code | Log bạn gửi xác nhận 9 test thành công trước khi thêm mã vẽ biểu đồ |
| Tạo crop test | Đã xác minh: 500 ảnh nguồn, 10.065 crop hợp lệ, loại 2.188 vùng |
| Benchmark trước | Đã xác minh: chấm đủ 10.065 crop |
| Phân tích trước | Đã đối chiếu báo cáo, bảng độ dài và xem 5 crop sai |
| Chuẩn bị train/validation | Đã xác minh: 25.723 crop train, 7.191 crop validation |
| Train và validation theo epoch | Đã xác minh: đủ 10 epoch, best ở epoch 9 trên 7.191 crop validation |
| Benchmark/phân tích sau | Chưa chạy |
| Biểu đồ | Đã thêm mã; chờ bạn chạy |

| Chỉ số cuối | Trước | Sau |
| --- | --- | --- |
| Số crop test | 10.065 | Chưa đo |
| Số chuỗi đúng / sai | 6.160 / 3.905 | Chưa đo |
| Exact Match | 61,20% | Chưa đo |
| CER | 20,68% (7.702 phép sửa / 37.246 ký tự đáp án) | Chưa đo |
| Mean / median / P95 | 40,57 / 36,70 / 76,00 ms/crop | Chưa đo |
| Sai thành đúng / đúng thành sai | — | Chưa đo |

Nguồn đã đối chiếu: [thống kê chuẩn bị test](output/crops/test/preparation.json) và [chỉ số pretrained](output/benchmark/before/metrics.json). Các vùng bị loại gồm 2.167 nhãn `###`, 18 nhãn rỗng và 3 tứ giác không hợp lệ. Điều kiện đo: CPU, 4 thread, batch 1, 3 lượt warmup, greedy; PyTorch 2.15.0+cpu, VietOCR 0.3.13.

### Phân tích pretrained đã xác minh

Nguồn: [báo cáo trước](output/analysis/before/report.md), [bảng độ dài](output/analysis/before/by_length.csv) và [từng mẫu sai](output/analysis/before/errors.csv).

| Nhóm sai khác chuỗi | Số mẫu | Tỷ lệ trong 3.905 mẫu sai |
| --- | --- | --- |
| Khác nội dung/khoảng trắng | 2.453 | 62,82% |
| Chỉ khác dấu/đ | 1.231 | 31,52% |
| Chỉ khác hoa/thường | 161 | 4,12% |
| Khác cả dấu và hoa/thường | 60 | 1,54% |

| Độ dài đáp án | Số mẫu | Exact Match | CER |
| --- | --- | --- | --- |
| 1 ký tự | 559 | 55,64% | 52,42% |
| 2–3 ký tự | 4.760 | 60,80% | 23,20% |
| 4–6 ký tự | 4.181 | 64,31% | 17,94% |
| 7+ ký tự | 565 | 47,08% | 21,01% |

Nhóm 7+ ký tự có Exact Match thấp nhất, còn nhóm 1 ký tự có CER cao nhất; hai chỉ số đo các khía cạnh khác nhau. Chênh lệch giữa các nhóm là mô tả trên test, chưa phải kết luận nguyên nhân hoặc chỉ dẫn chọn hyperparameter theo test.

Đã xem 5 crop: `1504_0002` (ĐỊA → DIA), `1518_0009` (ĐẠT → DAT), `1762_0009`, `1822_0021`, `1804_0007`. Hai ví dụ đầu có dấu/đ thể hiện trên ảnh nhưng dự đoán làm mất. Các ví dụ khác gồm crop dấu câu bị dự đoán lặp từ, URL chữ nhỏ bị đọc thành nội dung khác và crop chữ nhỏ/mờ. Đây là quan sát ở các ảnh đã xem, không đại diện nguyên nhân của toàn bộ lỗi.

### Dữ liệu train/validation đã xác minh

| Tập | Ảnh nguồn | Crop giữ lại | Vùng bị loại |
| --- | --- | --- | --- |
| Train | 1.200 | 25.723 | 9.371 |
| Validation | 300 | 7.191 | 1.546 |

Nguồn: [thống kê train](output/crops/train/preparation.json), [thống kê validation](output/crops/val/preparation.json). Train loại 9.300 nhãn `###`, 51 tứ giác không hợp lệ, 2 nhãn ngoài vocab và 18 nhãn rỗng. Validation loại 1.517 nhãn `###`, 10 tứ giác không hợp lệ và 19 nhãn rỗng.

### Kết quả huấn luyện đã xác minh

Nguồn: [lịch sử epoch](output/finetuning/history.csv), [checkpoint tốt nhất](output/finetuning/best_metrics.json) và [tổng kết hoàn tất](output/finetuning/summary.json). Full fine-tuning đủ 10 epoch trên CPU, giữ nguyên 25.723 crop train và 7.191 crop validation.

| Epoch | Train loss | Val loss | Exact Match validation | CER validation |
| --- | --- | --- | --- | --- |
| 0 — pretrained, chưa train | — | 1,5535 | 55,71% | 25,54% |
| 9 — best.pth | 0,9657 | 1,2104 | 77,32% | 12,66% |
| 10 — last.pth | 0,9559 | 1,2210 | 76,54% | 13,43% |

So với epoch 0 trên **cùng validation**, epoch 9 tăng Exact Match 21,61 điểm phần trăm và giảm CER 12,88 điểm phần trăm. Checkpoint này đọc đúng hoàn toàn 5.560/7.191 crop, còn 3.393 phép sửa trên 26.804 ký tự đáp án. Loss validation thấp nhất ở epoch 3 (1,1868), nhưng epoch 9 có điểm chuỗi tốt nhất theo tiêu chí đã định.

Train loss tiếp tục giảm đến epoch 10 trong khi val loss tăng và điểm chuỗi kém hơn epoch 9: có dấu hiệu chững lại/overfitting, chưa đủ để kết luận nên train thêm. Dùng `best.pth` đã chọn bằng validation để chấm test. Không lấy 77,32% validation trừ 61,20% test để báo mức cải thiện. Chưa có kết luận về hiệu quả trên test sau fine-tuning.

Bước tiếp theo: chạy `benchmark.run after`, rồi `analysis.report after` và `analysis.plots test`. README chỉ cập nhật kết quả test sau khi bạn chạy xong và có số liệu thực tế.

## Đưa code lên Git

Theo lựa chọn của bạn, đưa cả code và toàn bộ `output` lên repository, gồm crop, dự đoán, báo cáo, biểu đồ và checkpoint. `.gitignore` vẫn loại môi trường ảo, cache Python, dữ liệu gốc `vintext`/`vintext_original` và cấu hình riêng của máy. `requirements.txt` được commit để cài dependencies; không đưa thư viện trong `.venv` lên Git.

`.gitattributes` dùng Git LFS cho ảnh PNG và trọng số trong `output`. Git lưu con trỏ, Git LFS tải nội dung file lên máy chủ khi push; xem [tài liệu Git LFS](https://git-lfs.com/). Tại thời điểm kiểm tra, `output` có 43.016 file, tổng 969.750.498 byte (khoảng 0,90 GiB). Chưa commit hoặc push output thay bạn.

Repository đã được khởi tạo và có remote `https://github.com/NguyenThHien/OCRreg.git`. Không cần chạy lại `git init` hoặc `git remote add`. Trong PowerShell tại `D:\OCRreg`, chạy lần lượt:

```powershell
git lfs install --local
git add .gitignore .gitattributes README.md
git add output
git diff --cached --stat
```

Kiểm tra danh sách đã có file trong `output` và không có `.venv` hoặc dữ liệu gốc, rồi tiếp tục:

```powershell
git commit -m "Thêm output benchmark và fine-tuning với Git LFS"
git push origin main
```

Khi dùng máy khác đã cài Git LFS, chạy `git lfs install`, clone repository, rồi `git lfs pull` trong thư mục vừa clone để tải đủ ảnh/trọng số. Dữ liệu gốc và môi trường Python vẫn cần chuẩn bị riêng. LFS sử dụng dung lượng lưu trữ/băng thông của tài khoản GitHub; xem [cách tính dung lượng LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-storage-and-bandwidth-usage).
