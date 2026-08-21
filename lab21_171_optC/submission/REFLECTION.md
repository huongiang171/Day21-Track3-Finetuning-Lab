# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là việc tăng Rank (bản `attn_only` với rank rất cao) không hề bù đắp được cho việc gắn adapter sai vị trí (chỉ gắn vào `q,v`). Train loss giảm sâu đánh lừa cảm giác mô hình đang học tốt, nhưng thực tế điểm target không hề nhỉnh hơn bản chuẩn với rank cực nhỏ (16) nhưng gắn đúng chỗ.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở việc đồng bộ các file kết quả từ môi trường Colab về máy cá nhân (lỗi cache của Colab khiến tải về toàn file cũ). Khúc cấu hình pipeline chạy model thì mượt mà hơn tôi tưởng nhờ có script làm sẵn.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi nghĩ QLoRA (4-bit) luôn là phương pháp mặc định tối ưu nhất vì nó tiết kiệm VRAM. Giờ thì tôi không tin mù quáng nữa: bản `qlora` trên dòng Qwen3.5 này thực tế tốn thời gian train lâu hơn và làm giảm chất lượng đầu ra khá nhiều, không đáng để đánh đổi.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI để phân tích và giải thích cặn kẽ các thông số `train loss` vs `target` trong file `autopsy.json`. AI rất giỏi móc nối kiến thức nhưng thi thoảng nhầm lẫn một chút trong việc kiểm tra file thực tế trên ổ cứng do không nhận ra Colab đang không cập nhật file.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên là phải tạo ra bằng được một bộ *mặt nạ loss (loss mask)* thật chuẩn xác (chỉ tính loss trên câu trả lời, che đi câu hỏi) và thiết lập ngay một baseline không cần huấn luyện (chỉ dùng prompt) để xem thực sự mình đang cần cải thiện điều gì.
