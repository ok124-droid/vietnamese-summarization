import math
import torch
import torch.nn as nn
from typing import Optional

class MultiHeadAttention(nn.Module):
    """
    Cơ chế Chú ý Đa đầu (Multi-Head Attention - MHA) dùng trong mô hình Transformer.
    Hỗ trợ cả ba dạng:
    1. Encoder Self-Attention (Query = Key = Value = Encoder Outputs, mask là padding mask)
    2. Decoder Masked Self-Attention (Query = Key = Value = Decoder Outputs, mask là padding + causal mask)
    3. Cross-Attention / Encoder-Decoder Attention (Query = Decoder Output, Key = Value = Encoder Output)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) phải chia hết cho num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Số chiều của mỗi head
        
        # Các ma trận chiếu tuyến tính cho Query, Key, Value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Ma trận chiếu tuyến tính đầu ra sau khi ghép các đầu (concatenate)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.attention_dropout = nn.Dropout(p=dropout)
        
    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q: Tensor Query kích thước [batch_size, seq_len_q, d_model]
            k: Tensor Key kích thước [batch_size, seq_len_k, d_model]
            v: Tensor Value kích thước [batch_size, seq_len_k, d_model]
            mask: Tensor mặt nạ kích thước có thể là:
                  - 2D: [batch_size, seq_len_k] (padding mask)
                  - 3D: [batch_size, seq_len_q, seq_len_k] (padding + causal mask)
                  - 4D: [batch_size, 1, seq_len_q, seq_len_k]
                  (Mặt nạ có giá trị True tại vị trí cần che - padding/tương lai, False tại vị trí hợp lệ)
        Returns:
            Tensor đầu ra kích thước [batch_size, seq_len_q, d_model]
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        
        # 1. Chiếu tuyến tính Q, K, V qua các Layer Linear tương ứng
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        q_projected = self.q_linear(q)
        k_projected = self.k_linear(k)
        v_projected = self.v_linear(v)
        
        # 2. Tách và chuyển đổi chiều thành đa đầu (Multi-Head)
        # Thay đổi hình dạng: [batch_size, seq_len, num_heads, d_k]
        # Sau đó chuyển vị: [batch_size, num_heads, seq_len, d_k]
        q_heads = q_projected.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k_heads = k_projected.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v_heads = v_projected.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Tính toán Attention Scores (Q * K^T / sqrt(d_k))
        # k_heads.transpose(-2, -1) chuyển vị 2 chiều cuối: [batch_size, num_heads, d_k, seq_len_k]
        # Kết quả phép nhân ma trận: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4. Áp dụng Mask (Mặt nạ) nếu có
        if mask is not None:
            # Tự động định hình lại mask nếu truyền vào 2D hoặc 3D để broadcast tương thích với scores (4D)
            if mask.dim() == 2:
                # [batch_size, seq_len_k] -> [batch_size, 1, 1, seq_len_k]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # [batch_size, seq_len_q, seq_len_k] -> [batch_size, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)
            
            # Thay thế các vị trí có giá trị True bằng một số âm cực lớn (gần âm vô cùng)
            # để khi đi qua hàm softmax, xác suất chú ý (attention weight) tại đó tiến về 0.
            scores = scores.masked_fill(mask, float('-inf'))
            
        # 5. Softmax trên chiều cuối cùng để lấy phân phối xác suất chú ý (Attention Weights)
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Áp dụng Dropout lên các trọng số chú ý thu được
        attn_weights = self.attention_dropout(attn_weights)
        
        # 6. Nhân trọng số với Value vectors (Attention weights * V)
        # [batch_size, num_heads, seq_len_q, seq_len_k] x [batch_size, num_heads, seq_len_k, d_k]
        # Kết quả: [batch_size, num_heads, seq_len_q, d_k]
        context = torch.matmul(attn_weights, v_heads)
        
        # 7. Ghép các đầu lại (Concatenation)
        # Chuyển vị về dạng: [batch_size, seq_len_q, num_heads, d_k]
        # Sau đó gộp 2 chiều cuối: [batch_size, seq_len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # 8. Đi qua tầng chiếu Linear đầu ra cuối cùng
        # [batch_size, seq_len_q, d_model] -> [batch_size, seq_len_q, d_model]
        output = self.out_linear(context)
        
        return output
