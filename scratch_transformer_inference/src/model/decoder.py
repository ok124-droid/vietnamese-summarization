import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForward
from .embedding import TransformerEmbedding, TokenEmbedding

class DecoderLayer(nn.Module):
    """
    Một tầng Decoder đơn lẻ (Decoder Block) trong mô hình Transformer.
    Áp dụng cơ chế Pre-Layer Normalization (Pre-LN) tương tự như EncoderLayer.
    Cấu trúc gồm:
    1. Masked Self-Attention (chú ý trên các token đã sinh ở decoder)
    2. Cross-Attention (chú ý trên đầu ra của encoder - Encoder Output)
    3. Position-wise Feed-Forward Network
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        # 1. Tầng tự chú ý đa đầu có mặt nạ (Masked Self-Attention)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        
        # 2. Tầng chú ý chéo tương tác với Encoder (Cross-Attention)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        
        # 3. Tầng truyền thẳng (Feed-Forward)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, 
                x: torch.Tensor, 
                enc_out: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor biểu diễn decoder hiện tại, kích thước [batch_size, seq_len_tgt, d_model]
            enc_out: Tensor đầu ra của Encoder, kích thước [batch_size, seq_len_src, d_model]
            tgt_mask: Mặt nạ cho decoder (gồm causal mask và target padding mask) [batch_size, 1, seq_len_tgt, seq_len_tgt]
            src_mask: Mặt nạ padding của encoder (source padding mask) [batch_size, 1, 1, seq_len_src]
        Returns:
            Tensor biểu diễn sau tầng, kích thước [batch_size, seq_len_tgt, d_model]
        """
        # --- Phân nhánh 1: Masked Self-Attention kết hợp Pre-LN ---
        norm_x = self.norm1(x)
        self_attn_out = self.self_attn(q=norm_x, k=norm_x, v=norm_x, mask=tgt_mask)
        x = x + self.dropout1(self_attn_out)

        # --- Phân nhánh 2: Cross-Attention tương tác Encoder kết hợp Pre-LN ---
        # Query lấy từ Decoder (norm_x), Key và Value lấy từ Encoder (enc_out)
        norm_x = self.norm2(x)
        cross_attn_out = self.cross_attn(q=norm_x, k=enc_out, v=enc_out, mask=src_mask)
        x = x + self.dropout2(cross_attn_out)

        # --- Phân nhánh 3: Feed-Forward kết hợp Pre-LN ---
        norm_x = self.norm3(x)
        ffn_out = self.feed_forward(norm_x)
        x = x + self.dropout3(ffn_out)

        return x


class TransformerDecoder(nn.Module):
    """
    Bộ giải mã hoàn chỉnh (Transformer Decoder) gồm nhiều tầng DecoderLayer xếp chồng lên nhau.
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 num_layers: int, 
                 num_heads: int, 
                 d_ff: int, 
                 max_len: int = 96, # max_target_length trong config là 96
                 pad_id: int = 0, 
                 dropout: float = 0.1, 
                 activation: str = "gelu",
                 token_emb: Optional[TokenEmbedding] = None):
        super().__init__()
        self.pad_id = pad_id
        
        # Khởi tạo TransformerEmbedding cho Decoder, cho phép chia sẻ TokenEmbedding từ ngoài
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size, 
            d_model=d_model, 
            max_len=max_len, 
            pad_id=pad_id, 
            dropout=dropout,
            token_emb=token_emb
        )
            
        # Xếp chồng nhiều tầng Decoder Layer
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_ff, 
                dropout=dropout, 
                activation=activation
            ) for _ in range(num_layers)
        ])
        
        # LayerNorm cuối cùng cho đầu ra của cấu trúc Pre-LN
        self.final_norm = nn.LayerNorm(d_model)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Tạo mặt nạ tam giác trên (Causal Mask) ngăn decoder nhìn trước các từ ở tương lai.
        Trả về True ở các vị trí cần che (tương lai), False ở vị trí hợp lệ.
        """
        # torch.triu(..., diagonal=1) trả về ma trận tam giác trên không tính đường chéo chính
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self, 
                tokens: torch.Tensor, 
                enc_out: torch.Tensor, 
                tgt_pad_mask: Optional[torch.Tensor] = None, 
                src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: Tensor chứa Token IDs của decoder, kích thước [batch_size, seq_len_tgt]
            enc_out: Tensor biểu diễn từ Encoder, kích thước [batch_size, seq_len_src, d_model]
            tgt_pad_mask: Mặt nạ padding của decoder, kích thước [batch_size, seq_len_tgt] (True là padding)
            src_pad_mask: Mặt nạ padding của encoder, kích thước [batch_size, seq_len_src] (True là padding)
        Returns:
            Tensor biểu diễn ngữ nghĩa đầu ra của Decoder, kích thước [batch_size, seq_len_tgt, d_model]
        """
        batch_size, seq_len_tgt = tokens.size()
        device = tokens.device
        
        # 1. Đi qua tầng nhúng kết hợp mã hóa vị trí
        # [batch_size, seq_len_tgt] -> [batch_size, seq_len_tgt, d_model]
        x = self.embedding(tokens)
        
        # 2. Tạo Causal Mask và kết hợp với Target Padding Mask (nếu có)
        # causal_mask: [seq_len_tgt, seq_len_tgt]
        causal_mask = self.generate_causal_mask(seq_len_tgt, device)
        
        # Tự động tạo padding mask cho target nếu người dùng không truyền vào
        if tgt_pad_mask is None:
            tgt_pad_mask = tokens.eq(self.pad_id)
            
        # Mở rộng chiều của padding mask: [batch_size, 1, 1, seq_len_tgt]
        expanded_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)
        # Mở rộng chiều của causal mask: [1, 1, seq_len_tgt, seq_len_tgt]
        expanded_causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        # Kết hợp bằng phép OR logic: Nếu là token pad HOẶC là token ở tương lai -> Che đi (True)
        tgt_mask = expanded_pad_mask | expanded_causal_mask
            
        # 3. Chuẩn bị Source Mask (padding mask của encoder)
        if src_pad_mask is not None:
            # Mở rộng chiều để tương thích broadcast: [batch_size, 1, 1, seq_len_src]
            src_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)
        else:
            src_mask = None
            
        # 4. Đi qua các tầng Decoder Layer xếp chồng
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)
            
        # 5. Chuẩn hóa LayerNorm cuối cùng cho Pre-LN
        x = self.final_norm(x)
        
        return x
