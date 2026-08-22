import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .embedding import TransformerEmbedding, TokenEmbedding

class EncoderLayer(nn.Module):
    """
    Một tầng Encoder đơn lẻ (Encoder Block) trong mô hình Transformer.
    Áp dụng cơ chế Pre-Layer Normalization (Pre-LN) để tăng độ ổn định khi huấn luyện.
    Cấu trúc gồm:
    x -> LayerNorm -> Self-Attention -> Dropout -> + -> LayerNorm -> Feed-Forward -> Dropout -> +
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        # Tầng chú ý đa đầu tự chú ý (Self-Attention)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        
        # Tầng truyền thẳng (Feed-Forward Network)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor biểu diễn đầu vào, kích thước [batch_size, seq_len_q, d_model]
            mask: Tensor mặt nạ (padding mask) kích thước [batch_size, 1, 1, seq_len_k] hoặc [batch_size, seq_len_k]
        Returns:
            Tensor đại diện đầu ra sau tầng, kích thước [batch_size, seq_len_q, d_model]
        """
        # --- Phân nhánh 1: Self-Attention kết hợp Pre-LN ---
        # 1. Chuẩn hóa trước (Pre-LN)
        norm_x = self.norm1(x)
        # 2. Tính toán Self-Attention (Query = Key = Value = norm_x)
        attn_out = self.self_attn(q=norm_x, k=norm_x, v=norm_x, mask=mask)
        # 3. Cộng kết nối tắt (Residual Connection) sau khi áp dụng Dropout
        x = x + self.dropout1(attn_out)

        # --- Phân nhánh 2: Position-wise Feed-Forward kết hợp Pre-LN ---
        # 1. Chuẩn hóa trước (Pre-LN)
        norm_x = self.norm2(x)
        # 2. Truyền qua tầng mạng phi tuyến Feed-Forward
        ffn_out = self.feed_forward(norm_x)
        # 3. Cộng kết nối tắt (Residual Connection) sau khi áp dụng Dropout
        x = x + self.dropout2(ffn_out)

        return x


class TransformerEncoder(nn.Module):
    """
    Bộ mã hóa hoàn chỉnh (Transformer Encoder) gồm nhiều tầng EncoderLayer xếp chồng lên nhau.
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 num_layers: int, 
                 num_heads: int, 
                 d_ff: int, 
                 max_len: int = 768, 
                 pad_id: int = 0, 
                 dropout: float = 0.1, 
                 activation: str = "gelu",
                 token_emb: Optional[TokenEmbedding] = None):
        super().__init__()
        
        # Khởi tạo TransformerEmbedding cho Encoder, cho phép chia sẻ TokenEmbedding từ ngoài
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size, 
            d_model=d_model, 
            max_len=max_len, 
            pad_id=pad_id, 
            dropout=dropout,
            token_emb=token_emb
        )
            
        # Xếp chồng nhiều tầng Encoder Layer
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_ff, 
                dropout=dropout, 
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        # LayerNorm cuối cùng bắt buộc phải có khi sử dụng cấu trúc Pre-LN
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: Tensor chứa Token IDs đầu vào, kích thước [batch_size, seq_len]
            mask: Tensor mặt nạ padding, kích thước [batch_size, seq_len] hoặc đã định dạng lại
        Returns:
            Tensor biểu diễn ngữ nghĩa trích xuất (Encoder Hidden States) kích thước [batch_size, seq_len, d_model]
        """
        # 1. Đi qua tầng nhúng kết hợp mã hóa vị trí
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.embedding(tokens)
        
        # 2. Đi qua lần lượt từng tầng EncoderLayer xếp chồng
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        # 3. Áp dụng chuẩn hóa LayerNorm cuối cùng cho đầu ra
        x = self.final_norm(x)
        
        return x
