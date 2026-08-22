import torch
import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    """
    Tầng Feed-Forward theo vị trí (Position-wise Feed-Forward Network - FFN).
    Được áp dụng riêng biệt cho từng vị trí (từng token) sau tầng Multi-Head Attention.
    Cấu trúc gồm: Linear -> Activation -> Dropout -> Linear.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # Chọn hàm kích hoạt dựa trên cấu hình (mặc định trong config là gelu)
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Hàm kích hoạt '{activation}' không được hỗ trợ. Hãy chọn 'gelu' hoặc 'relu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor đầu vào từ tầng chú ý trước đó, kích thước [batch_size, seq_len, d_model]
        Returns:
            Tensor đầu ra sau biến đổi phi tuyến tính, kích thước [batch_size, seq_len, d_model]
        """
        # Bước 1: Chiếu tuyến tính lên không gian ẩn lớn hơn (d_ff) và áp dụng hàm phi tuyến
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_ff]
        x = self.activation(self.w_1(x))
        
        # Bước 2: Áp dụng Dropout
        x = self.dropout(x)
        
        # Bước 3: Chiếu tuyến tính ngược lại không gian ban đầu (d_model)
        # [batch_size, seq_len, d_ff] -> [batch_size, seq_len, d_model]
        x = self.w_2(x)
        
        return x
