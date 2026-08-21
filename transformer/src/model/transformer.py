import torch
import torch.nn as nn
from typing import Optional
from .embedding import TokenEmbedding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

class TransformerSeq2Seq(nn.Module):
    """
    Mô hình Transformer Sequence-to-Sequence (Encoder-Decoder) hoàn chỉnh.
    Được tối ưu hóa cho bài toán tóm tắt văn bản (Abstractive Summarization) tiếng Việt.
    Hỗ trợ cấu hình linh hoạt từ file YAML (Pre-LN, Weight Tying, Dynamic padding).
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int, 
                 num_heads: int, 
                 d_ff: int, 
                 max_source_length: int = 768, 
                 max_target_length: int = 96, 
                 pad_id: int = 0, 
                 dropout: float = 0.1, 
                 activation: str = "gelu",
                 share_encoder_decoder_embeddings: bool = True,
                 tie_embeddings: bool = True):
        super().__init__()
        
        self.pad_id = pad_id
        
        # 1. Khởi tạo tầng Token Embedding
        if share_encoder_decoder_embeddings:
            # Hai bộ Encoder và Decoder dùng chung lớp Token Embedding (chia sẻ trọng số biểu diễn từ vựng)
            self.shared_token_emb = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)
            encoder_token_emb = self.shared_token_emb
            decoder_token_emb = self.shared_token_emb
        else:
            # Khởi tạo riêng biệt nếu không cấu hình chia sẻ
            self.shared_token_emb = None
            encoder_token_emb = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)
            decoder_token_emb = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)

        # 2. Khởi tạo bộ mã hóa (Transformer Encoder)
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_source_length,
            pad_id=pad_id,
            dropout=dropout,
            activation=activation,
            token_emb=encoder_token_emb
        )

        # 3. Khởi tạo bộ giải mã (Transformer Decoder)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_target_length,
            pad_id=pad_id,
            dropout=dropout,
            activation=activation,
            token_emb=decoder_token_emb
        )

        # 4. Tầng tuyến tính chiếu đầu ra (Output Projection) để tính xác suất phân phối từ vựng
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # 5. Áp dụng cơ chế chia sẻ trọng số Weight Tying (nếu được cấu hình)
        if tie_embeddings:
            # Trọng số của tầng Linear đầu ra sẽ dùng chung địa chỉ bộ nhớ với Target Token Embedding của Decoder
            self.output_projection.weight = self.decoder.embedding.token_emb.embedding.weight

    def encode(self, src_tokens: torch.Tensor, src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Mã hóa chuỗi đầu vào (Source Tokens) thành các vector biểu diễn ngữ nghĩa.
        Tiện lợi khi gọi riêng biệt trong quá trình sinh văn bản (Inference/Decode).
        """
        if src_pad_mask is not None:
            src_pad_mask = src_pad_mask.to(device=src_tokens.device, dtype=torch.bool)
        return self.encoder(src_tokens, mask=src_pad_mask)

    def decode(self, 
               tgt_tokens: torch.Tensor, 
               enc_out: torch.Tensor, 
               tgt_pad_mask: Optional[torch.Tensor] = None, 
               src_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Giải mã dựa trên trạng thái ngữ nghĩa từ Encoder và các token đã sinh ở Decoder.
        """
        if tgt_pad_mask is not None:
            tgt_pad_mask = tgt_pad_mask.to(device=tgt_tokens.device, dtype=torch.bool)
        if src_pad_mask is not None:
            src_pad_mask = src_pad_mask.to(device=tgt_tokens.device, dtype=torch.bool)
            
        # Trả về các hidden states của Decoder [batch_size, seq_len_tgt, d_model]
        dec_out = self.decoder(
            tokens=tgt_tokens, 
            enc_out=enc_out, 
            tgt_pad_mask=tgt_pad_mask, 
            src_pad_mask=src_pad_mask
        )
        # Chiếu qua tầng Output Projection để lấy logit phân bố từ vựng [batch_size, seq_len_tgt, vocab_size]
        logits = self.output_projection(dec_out)
        return logits

    def forward(self, 
                src_tokens: torch.Tensor, 
                tgt_tokens: torch.Tensor, 
                src_pad_mask: Optional[torch.Tensor] = None, 
                tgt_pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Quy trình Lan truyền xuôi (Forward Pass) trong quá trình huấn luyện:
        Args:
            src_tokens: IDs đầu vào của Encoder, kích thước [batch_size, seq_len_src]
            tgt_tokens: IDs đầu vào của Decoder, kích thước [batch_size, seq_len_tgt]
            src_pad_mask: Mặt nạ padding của Encoder, kích thước [batch_size, seq_len_src]
            tgt_pad_mask: Mặt nạ padding của Decoder, kích thước [batch_size, seq_len_tgt]
        Returns:
            Logits phân phối từ vựng trên mỗi bước dịch của decoder [batch_size, seq_len_tgt, vocab_size]
        """
        # 1. Mã hóa đầu vào
        enc_out = self.encode(src_tokens, src_pad_mask=src_pad_mask)
        
        # 2. Giải mã và chiếu ra logits
        logits = self.decode(
            tgt_tokens=tgt_tokens, 
            enc_out=enc_out, 
            tgt_pad_mask=tgt_pad_mask, 
            src_pad_mask=src_pad_mask
        )
        
        return logits
