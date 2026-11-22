# Báo cáo Thực nghiệm Đối chuẩn: Semantic Search CLIP vs SigLIP trên MS COCO và NewsSeek

---

## 1. Giới thiệu & Phạm vi Thử nghiệm

Báo cáo này lưu trữ và tổng hợp kết quả đối chuẩn (benchmark) thực nghiệm chi tiết giữa hai mô hình thị giác - ngôn ngữ nền tảng:
* **OpenAI CLIP (ViT-B/32)**: Không gian nhúng 512D Float32, Patch Size 32x32, Tokenizer BPE (vocab 49,408).
* **Google SigLIP (ViT-B/16-224)**: Không gian nhúng 768D Float32, Patch Size 16x16, Tokenizer WebLI SentencePiece (vocab 250,000).

Thử nghiệm được thực hiện trên hai kho ngữ liệu đại diện cho hai bài toán truy xuất đa phương thức:
1. **MS COCO Karpathy Test Split**: 5,000 ảnh tĩnh chuẩn hóa, 25,010 câu truy vấn chú thích đối xứng.
2. **NewsSeek Multilingual Video Dataset**: 698 video thời sự đa ngôn ngữ, 164,476 khung hình tiêu biểu (keyframes) được trích xuất theo scene boundaries.

---

## 2. Bảng Ma trận Số liệu Định lượng (Quantitative Benchmark Matrix)

### 2.1. Đánh giá trên MS COCO Karpathy Test Split (5K Images / 25,010 Captions)

| Chỉ số Đánh giá | OpenAI CLIP (ViT-B/32) | Google SigLIP (ViT-B/16-224) | Chênh lệch (Delta) | Mô hình Vượt trội |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@1 (Text-to-Image)** | 30.51% | **47.81%** | +17.30% | SigLIP |
| **Recall@5 (Text-to-Image)** | 55.96% | **72.45%** | +16.49% | SigLIP |
| **Recall@10 (Text-to-Image)** | 66.82% | **80.98%** | +14.16% | SigLIP |
| **Mean Recall (mR)** | 51.10% | **67.08%** | +15.98% | SigLIP |
| **Mean Reciprocal Rank (MRR)**| 0.4262 | **0.5908** | +0.1646 | SigLIP |
| **Modality Gap** | 0.8269 | 1.0524 | +0.2255 | CLIP (Khoảng cách hẹp hơn) |
| **Điểm tương đồng trung bình (Positive)** | 0.3046 | 0.1150 | -0.1896 | CLIP (Do nhiệt độ t cao hơn) |
| **Thời gian mã hóa 5,000 ảnh (RTX 4060)** | 888.07 s (5.6 img/s) | **50.46 s (99.1 img/s)** | **17.6x Nhanh hơn** | SigLIP |
| **Thời gian mã hóa 25,010 văn bản** | 7.20 s | 24.26 s | +17.06 s | CLIP |

---

### 2.2. Đánh giá trên NewsSeek Benchmark (698 Videos / 164,476 Keyframes)

| Chỉ số Đánh giá | OpenAI CLIP (ViT-B/32) | Google SigLIP (ViT-B/16-224) | Chênh lệch (Delta) | Mô hình Vượt trội |
| :--- | :--- | :--- | :--- | :--- |
| **Recall@1 (Video Max-Pool)** | 38.40% | **40.54%** | +2.14% (+5.6% tương đối) | SigLIP |
| **Recall@5 (Video Max-Pool)** | 63.47% | **64.61%** | +1.14% | SigLIP |
| **Recall@10 (Video Max-Pool)** | **75.64%** | 73.78% | -1.86% | CLIP |
| **Mean Recall (mR)** | 59.17% | **59.64%** | +0.47% | SigLIP |
| **Mean Reciprocal Rank (MRR)** | 0.5009 | **0.5159** | +0.0150 | SigLIP |
| **R@1 English (615 video)** | 39.19% | **42.28%** | +3.09% | SigLIP |
| **R@1 Non-English (83 video)** | **32.53%** | 27.71% | -4.82% | CLIP |
| **Thông lượng mã hóa (Colab GPU)** | 57.3 fps | 57.3 fps | Song song FP16 In-Memory | Đồng hạng |
| **Tổng thời gian mã hóa (164,476 frames)** | 2,870.93 s (~47.8 phút) | 2,870.93 s (~47.8 phút) | Sequential grab + Prefetch | Đồng hạng |
| **Modality Gap** | **0.8230** | 1.0622 | +0.2392 | CLIP |

---

## 3. Phân tích Chi tiết Hiệu năng Truy xuất Ngữ nghĩa

### 3.1. Sự vượt trội của SigLIP trên Ảnh Tĩnh (MS COCO)
Trên tập MS COCO, SigLIP đạt mức bứt phá áp đảo với +17.30% R@1 (47.81% so với 30.51%) và MRR tăng từ 0.4262 lên 0.5908:
* **Độ phân giải Patch 16x16**: Với cùng kích thước ảnh đầu vào $224 \times 224$, SigLIP phân chia ảnh thành $14 \times 14 = 196$ patch tokens, trong khi CLIP ViT-B/32 chỉ có $7 \times 7 = 49$ patch tokens. Số lượng visual tokens nhiều gấp 4 lần cho phép SigLIP nhận diện các vật thể nhỏ (ví dụ: quả bóng tennis, chiếc nĩa, biển báo giao thông xa) mà CLIP thường bỏ sót.
* **Không gian biểu diễn 768D**: Chiều không gian nhúng lớn hơn (+256 chiều so với 512D của CLIP) giúp lưu trữ các thuộc tính thuộc về màu sắc, chất liệu và quan hệ không gian mà không bị nén mất thông tin.

### 3.2. Hiệu năng Truy xuất Video Tin tức Đa ngôn ngữ (NewsSeek)
* **Dẫn đầu ở độ chính xác Top-1, Top-5 và MRR**: Trên toàn bộ 698 video (164,476 keyframes), SigLIP đạt **40.54% R@1** (+2.14% so với 38.40% của CLIP), **64.61% R@5** (+1.14%) và **0.5159 MRR** (so với 0.5009), đặc biệt nổi bật trên tập 615 video tiếng Anh (**42.28% vs 39.19%**).
* **Phân tích hiện tượng R@10 và nhóm Non-EN (83 video)**: CLIP đạt R@10 cao hơn nhẹ (75.64% vs 73.78%) và R@1_NonEN đạt 32.53% so với 27.71% của SigLIP. Khảo sát metadata `videos.csv` cho thấy một tỷ lệ lớn video gắn nhãn `vi` (VTV24) và `ko` thực chất sử dụng tiêu đề dịch sang tiếng Anh, trong khi các tiêu đề `ja`/`zh` chứa nhiều tên riêng địa danh và ngày phát sóng đặc thù.
* **Chiến lược Max-Pooling & Hiệu năng Streaming**: Lấy điểm tương đồng cực đại trên các keyframe theo từng cảnh giúp lọc nhiễu chuyển cảnh và tập trung vào khoảnh khắc sự kiện chính. Pipeline giải mã tuần tự (`cap.grab()`) kết hợp `ThreadPoolExecutor` prefetch rút ngắn tổng thời gian encode cả hai mô hình xuống **47.8 phút (57.3 fps)**.

---

## 4. Kết luận Thực nghiệm
Kết quả thực nghiệm trên cả hai tập ngữ liệu khẳng định Google SigLIP vượt trội hơn OpenAI CLIP về chất lượng xếp hạng đỉnh (R@1, R@5, MRR, mR) trên cả ảnh tĩnh chuẩn tắc (MS COCO: +17.30% R@1) lẫn tập video thời sự thực tế (NewsSeek: +2.14% R@1, +0.0150 MRR), trong khi CLIP giữ được khoảng cách Modality Gap hẹp hơn và độ phủ R@10 cạnh tranh trên video tin tức.
