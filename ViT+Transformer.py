import json
import math
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from tqdm import tqdm
from datetime import datetime, timedelta

from datasets import ImageTextDataset
from vocabulary import Vocabulary


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, emb_size: int, img_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # Shape: (batch_size, emb_size, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)  # Shape: (batch_size, emb_size, n_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, n_patches, emb_size)
        return x


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, expansion: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, expansion * emb_size)
        self.fc2 = nn.Linear(expansion * emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, expansion: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout)
        self.feed_forward = FeedForward(emb_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention, _ = self.attention(x, x, x, mask)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        return x


class VisionTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, img_size: int, emb_size: int, num_layers: int, num_heads: int,
                 expansion: int, dropout: float):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positional_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.n_patches, emb_size))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(emb_size, num_heads, expansion, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.patch_embedding(x)
        batch_size, _, _ = x.shape
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_embedding
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x


def create_masks(target_seq, pad_idx, num_heads):
    # 创建三角掩码
    seq_len = target_seq.size(1)
    triangular_mask = torch.triu(torch.ones((seq_len, seq_len), device=target_seq.device) * float('-inf'), diagonal=1)

    # 创建PAD掩码
    pad_mask = (target_seq == pad_idx).to(target_seq.device)  # [batch_size, seq_len]
    pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, seq_len]

    # 合并掩码
    tgt_mask = triangular_mask.unsqueeze(0).expand(pad_mask.size(0), -1, -1)  # [batch_size, seq_len, seq_len]
    tgt_mask = tgt_mask.masked_fill(pad_mask, float('-inf'))

    # 调整掩码形状以适应多头注意力
    tgt_mask = tgt_mask.repeat_interleave(num_heads, dim=0)  # [batch_size * num_heads, seq_len, seq_len]

    return tgt_mask


class DecoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)
        self.self_attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.encoder_attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(emb_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # Self Attention
        x = x.transpose(0, 1)  # Change shape to [seq_length, batch_size, emb_size]
        enc_out = enc_out.transpose(0, 1)

        attention_output, _ = self.self_attention(x, x, x, attn_mask=trg_mask)
        query = self.dropout(self.norm1(attention_output + x))

        # Encoder-Decoder Attention
        attention_output, _ = self.encoder_attention(query, enc_out, enc_out, attn_mask=src_mask)
        query = self.dropout(self.norm2(attention_output + query))

        # Change shape back to [batch_size, seq_length, emb_size]
        query = query.transpose(0, 1)

        # Feed Forward
        out = self.feed_forward(query)
        out = self.dropout(self.norm3(out + query))

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Inject position encoding"""
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, emb_size, num_heads, expansion, dropout, num_layers, target_vocab_size,
                 pretrained_embeddings=None):
        super(TransformerDecoder, self).__init__()
        self.emb_size = emb_size
        self.word_embedding = nn.Embedding(target_vocab_size, emb_size)

        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (target_vocab_size, emb_size), "预训练嵌入向量尺寸不匹配"
            self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=False)
        else:
            self.word_embedding = nn.Embedding(target_vocab_size, emb_size)

        self.positional_encoding = PositionalEncoding(emb_size)
        self.layers = nn.ModuleList([DecoderLayer(emb_size, num_heads, expansion, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(emb_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.dropout(self.word_embedding(x))
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class ImageCaptioningModel(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, emb_size, target_vocab_size, num_layers, num_heads, expansion, dropout, pretrained_embeddings=None):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = VisionTransformerEncoder(in_channels=in_channels,
                                                patch_size=patch_size,
                                                img_size=img_size,
                                                emb_size=emb_size,
                                                num_layers=num_layers,
                                                num_heads=num_heads,
                                                expansion=expansion,
                                                dropout=dropout)
        self.decoder = TransformerDecoder(emb_size=emb_size,
                                          num_heads=num_heads,
                                          expansion=expansion,
                                          dropout=dropout,
                                          num_layers=num_layers,
                                          target_vocab_size=target_vocab_size,
                                          pretrained_embeddings=pretrained_embeddings)

    def forward(self, images, captions, src_mask=None, tgt_mask=None):
        """

        :param images: [batch_size, in_channels, img_size, img_size]
        :param captions: [seq_length, batch_size]
        """
        encoder_output = self.encoder(images)  # [batch_size, n_patches + 1, emb_size]
        decoder_output = self.decoder(captions, encoder_output, src_mask, tgt_mask)
        return decoder_output  # [seq_length, batch_size, target_vocab_size]


def train(model, train_loader, criterion, optimizer, mask_func, save_path, device, epochs=10,
          save_interval=120, pretrained_weights=None, experiment_name='experiment'):
    writer = SummaryWriter(f'runs/{experiment_name}')

    os.makedirs(os.path.split(save_path)[0], exist_ok=True)

    if pretrained_weights:
        model.load_state_dict(torch.load(pretrained_weights))

    # training loop
    p_bar = tqdm(range(epochs))
    model = model.to(device)
    save_interval = timedelta(seconds=save_interval)
    model.train()

    for epoch in p_bar:
        running_loss = 0.0

        last_save_time = datetime.now()
        for batch_idx, (image, seq, seq_len) in enumerate(train_loader):
            image = image.to(device)  # (batch, c, img_sz, img_sz)
            seq = seq.to(device)  # (batch, seq_len + 1)

            input_seq = seq[:, :-1]  # (batch, seq_len)
            target_seq = seq[:, 1:]  # (batch, seq_len)

            # 开始训练
            optimizer.zero_grad()
            tgt_mask = mask_func(input_seq)

            prediction = model(image, input_seq, tgt_mask=tgt_mask)  # (batch, seq_len, vocabulary_size)
            batch_size, _, vocab_size = prediction.shape
            loss = criterion(prediction.view(-1, vocab_size), target_seq.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            # autosave
            if datetime.now() - last_save_time > save_interval:
                last_save_time = datetime.now()
                torch.save(model.state_dict(), save_path)

            # 记录结果
            running_loss += loss.item()
            p_bar.set_postfix(progress=f'{(batch_idx + 1)} / {len(train_loader)}',
                              loss=f'{running_loss / (batch_idx + 1):.4f}',
                              last_save_time=last_save_time)
            writer.add_scalar('Loss/train', running_loss / (batch_idx + 1), epoch * len(train_loader) + batch_idx)

    writer.close()


def predict(model, image, transform, vocab, device, max_length, mask_func):
    model.to(device)
    image = transform(image).unsqueeze(0).to(device)  # -> (1, channel, img_size, img_size)
    seq = [vocab.start]

    # 将图像和文本序列放入设备（例如 GPU）
    image = image.to(device)
    seq = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_length):
        with torch.no_grad():
            tgt_mask = mask_func(seq)
            output = model(image, seq, tgt_mask=tgt_mask)

        predicted = output.argmax(2)[:, -1]
        seq = torch.cat((seq, predicted.unsqueeze(1)), dim=1)

        if predicted.item() == vocab.end:
            break

    # 将序列转换为单词列表'
    seq = seq.squeeze(0).cpu().numpy().tolist()
    text = vocab.decode(seq)

    return text


def validate(model, val_loader, criterion, optimizer, mask_func, save_path, device):
    pass


if __name__ == "__main__":
    # =================== Example parameters for the model ===================
    img_size = 224  # Size of the input image
    in_channels = 3  # Number of input channels (for RGB images)
    patch_size = 16  # Size of each patch
    emb_size = 96  # Embedding size
    num_layers = 6  # Number of layers in both encoder and decoder
    num_heads = 8  # Number of attention heads
    expansion = 4  # Expansion factor for feed forward network

    # =================== Train Config ===================
    dropout = 0.1  # Dropout rate
    lr = 5e-4  # Learning rate
    epochs = 500
    batch_size = 64  # Batch size
    seq_length = 128  # Max length of the caption sequence
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_path = 'models/model_transformer.pth'
    experiment_name = 'fashion_description'

    # =================== Vocabulary and Image transforms ===================
    vocabulary = Vocabulary('vocab.json')
    transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_func = partial(create_masks, pad_idx=vocabulary.pad, num_heads=num_heads)

    # =================== Initialize the model ===================
    model = ImageCaptioningModel(img_size, in_channels, patch_size, emb_size, len(vocabulary), num_layers, num_heads,
                                 expansion, dropout,
                                 pretrained_embeddings=vocabulary.get_word2vec())

    model.load_state_dict(torch.load('models/model_transformer.pth'))

    # =================== Prepare for Training ===================

    dataset = ImageTextDataset('data/deepfashion-multimodal',
                               'data/deepfashion-multimodal/train_captions.json',
                               vocabulary=vocabulary,
                               max_seq_len=seq_length,
                               transform=transform,
                               max_cache_memory=32 * 1024 ** 3)

    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # =================== Start Training ===================
    # train(model, dataloader, criterion, optimizer, mask_func, save_path='models/model_transformer.pth',
    #       epochs=epochs, device=device, experiment_name=experiment_name)

    #  =================== Model Inference ===================
    image_path, caption = dataset.sample()

    caption_generated = predict(model, Image.open(image_path), transform, vocabulary, device, seq_length, mask_func)
    print(caption, '\n\n', caption_generated)
