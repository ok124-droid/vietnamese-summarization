import math
import torch
import torch.nn as nn
from typing import Optional

class TokenEmbedding(nn.Module):
    """
    Lớp Embedding chuyển đổi các Token ID thành các Vector liên tục (dense vectors).
    Nhân với căn bậc hai của d_model (math.sqrt(d_model)) theo thiết kế của bài báo Attention Is All You Need.
    """
    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor chứa Token IDs có kích thước [batch_size, seq_len]
        Returns:
            Tensor embedding đã nhân tỉ lệ [batch_size, seq_len, d_model]
        """
        # Ép kiểu long phòng trường hợp dữ liệu đầu vào có kiểu dữ liệu khác
        return self.embedding(tokens.long()) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Lớp mã hóa vị trí dạng hình sin (Sinusoidal Positional Encoding).
    Cung cấp thông tin về thứ tự từ trong câu cho mô hình Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 768, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Khởi tạo ma trận mã hóa vị trí PE kích thước [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Cột chỉ số vị trí (pos) kích thước [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Số hạng chia (div_term) cho các tần số hình sin, cosin
        # Công thức: div_term = 10000^(2i / d_model)
        # Sử dụng log space để tính toán ổn định hơn: exp(2i * -log(10000) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # Áp dụng hàm Sin cho các vị trí chẵn (2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Áp dụng hàm Cos cho các vị trí lẻ (2i + 1)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Thêm chiều batch để có dạng [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Đăng ký pe như một 'buffer' thay vì 'parameter'.
        # Buffer sẽ được lưu trữ cùng trạng thái mô hình và tự động di chuyển sang GPU (cuda) cùng mô hình,
        # nhưng sẽ KHÔNG được tối ưu hóa (không cập nhật trọng số khi train).
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor embedding có kích thước [batch_size, seq_len, d_model]
        Returns:
            Tensor đã được cộng thêm thông tin vị trí và qua dropout [batch_size, seq_len, d_model]
        """
        # Cộng giá trị positional encoding tương ứng với độ dài chuỗi đầu vào (seq_len)
        # x.size(1) là seq_len của batch hiện tại (dynamic length)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Module kết hợp TokenEmbedding và PositionalEncoding.
    Tiện lợi cho việc sử dụng ở cả phía Encoder và Decoder.
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 768, pad_id: int = 0, dropout: float = 0.1, token_emb: Optional[TokenEmbedding] = None):
        super().__init__()
        if token_emb is not None:
            self.token_emb = token_emb
        else:
            self.token_emb = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor chứa Token IDs kích thước [batch_size, seq_len]
        Returns:
            Tensor đại diện đầy đủ (gồm từ vựng + vị trí) kích thước [batch_size, seq_len, d_model]
        """
        x = self.token_emb(tokens)
        x = self.pos_emb(x)
        return x
