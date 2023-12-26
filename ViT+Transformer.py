"""
ViT+Transformer for Image Captioning

This module implements an image captioning model combining Vision Transformer (ViT) and
Transformer. The ViT serves as an image encoder, converting input images into a sequence of
embedding vectors. These vectors are then fed into a Transformer decoder to generate
descriptive captions. The model supports custom positional embeddings, attention mechanisms,
and various utilities for image-text processing.

This script is a Python implementation of the 'ViT+Transformer.ipynb' Jupyter notebook,
adapted for standalone use and further development.

Author: zhanghanmo2021213368, hammershock
Date: 2023/10/26
License: MIT License
Usage: This file is part of the Project FashionDescription, specifically designed for
       tasks involving image captioning with ViT and Transformer architecture.

Dependencies:
- PyTorch: for model construction, training, and inference
- PIL: for image processing
- Matplotlib: for visualization
- Torchvision: for image transformations
- tqdm: for progress bar visualization
- Other custom modules: datasets.py and vocabulary.py for data handling and text processing
"""

import math
import os
from functools import partial
from typing import List, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from tqdm import tqdm
from datetime import datetime, timedelta

from datasets import ImageTextDataset
from vocabulary import Vocabulary
from metrics import bleu, meteor, rouge_l


class PatchEmbedding(nn.Module):
    """ViT嵌入层，通过将原始图像分为若干个小块，分别嵌入，然后展平为序列"""

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
    """编码器、解码器点对点前馈层"""

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
    """ViT编码器层"""

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
    """ViT编码器"""

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
    """创建目标序列注意力掩码"""
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
    """transformer解码器层"""

    def __init__(self, emb_size, num_heads, expansion, dropout):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)
        self.self_attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.encoder_attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.feed_forward = FeedForward(emb_size, expansion, dropout)
        self.dropout = nn.Dropout(dropout)
        self.encoder_decoder_att = None  # (batch, seq_len, image_embed_size)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # Self Attention
        x = x.transpose(0, 1)  # Change shape to [seq_length, batch_size, emb_size]
        enc_out = enc_out.transpose(0, 1)

        attention_output, _ = self.self_attention(x, x, x, attn_mask=trg_mask)
        query = self.dropout(self.norm1(attention_output + x))

        # Encoder-Decoder Attention
        attention_output, self.encoder_decoder_att = self.encoder_attention(query, enc_out, enc_out, attn_mask=src_mask)
        # print(self.encoder_decoder_att.shape)  # (batch, seq_len, image_embed_size)
        query = self.dropout(self.norm2(attention_output + query))

        # Change shape back to [batch_size, seq_length, emb_size]
        query = query.transpose(0, 1)

        # Feed Forward
        out = self.feed_forward(query)
        out = self.dropout(self.norm3(out + query))

        return out


class PositionalEncoding(nn.Module):
    """目标序列正余弦位置编码"""

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
    """Transformer解码器层"""

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
    """img2seq模型"""

    def __init__(self, img_size, in_channels, patch_size, emb_size, target_vocab_size, num_layers, num_heads, expansion,
                 dropout, pretrained_embeddings=None):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = VisionTransformerEncoder(in_channels=in_channels,
                                                patch_size=patch_size,
                                                img_size=img_size,
                                                emb_size=emb_size,
                                                num_layers=num_layers,
                                                num_heads=num_heads,
                                                expansion=expansion,
                                                dropout=dropout)
        self.decoder = TransformerDecoder(emb_size=emb_size,  # 简单起见，编码器图像块与解码器文本嵌入使用相同的嵌入维度
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

    def visualize(self):
        att_weights = self.decoder.layers[-1].encoder_decoder_att
        if att_weights is not None:
            return att_weights[:, -1, 1:]


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


def generate_by_beam_search(model, image, vocab, device, max_length, mask_func, beam_width=5):
    model = model.to(device)
    image = image.to(device)  # (1, channel, img_size, img_size)

    # 初始候选序列和分数
    sequences = [([vocab.start], 0)]  # 每个序列是(token_list, score)
    attn_images = []
    model.eval()
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == vocab.end:
                all_candidates.append((seq, score))
                continue

            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
            tgt_mask = mask_func(seq_tensor)

            with torch.no_grad():
                output = model(image, seq_tensor, tgt_mask=tgt_mask)
                attn_images.append((model.visualize(), vocab.inv[seq_tensor[:, -1].item()]))
                # 抽取最后一个decoder层、第一个batch、序列最后一个的交叉注意力权重

            # 考虑top k个候选
            top_k_probs, top_k_indices = torch.topk(torch.softmax(output, dim=-1), beam_width, dim=-1)
            top_k_probs = top_k_probs[0, -1]
            top_k_indices = top_k_indices[0, -1]

            for k in range(beam_width):
                next_seq = seq + [top_k_indices[k].item()]
                next_score = score - torch.log(top_k_probs[k])  # 使用负对数似然作为分数
                all_candidates.append((next_seq, next_score))

        # 按分数排序并选出前k个
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

    # 选择分数最高的序列
    best_seq = sequences[0][0]
    best_score = sequences[0][1].item()
    text = vocab.decode(best_seq)
    return text, best_score, attn_images


def visualize_attention(original_image, attn_maps: List[Tuple[torch.Tensor, str]], img_size=224, patch_size=16, max_cols=4):
    num_patches_side = img_size // patch_size
    num_steps = len(attn_maps)
    num_cols = min(num_steps, max_cols)
    num_rows = math.ceil(num_steps / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols + 1, figsize=(15, 6 * num_rows))

    for row in range(num_rows):
        # 显示原始图像
        axs[row, 0].imshow(original_image)
        axs[row, 0].axis('off')
        axs[row, 0].set_title('Original Image')

    for idx, (attn_image, current_word) in enumerate(attn_maps):
        # 转换注意力权重为热力图
        attention_image = attn_image.reshape(num_patches_side, num_patches_side).cpu().detach().numpy()

        # 将热力图大小调整为原始图像大小
        resized_attention = np.array(Image.fromarray(attention_image).resize(original_image.size, Image.BILINEAR))

        row = idx // num_cols
        col = idx % num_cols + 1

        # 在网格中显示注意力热力图
        im = axs[row, col].imshow(resized_attention, cmap='hot',
                                  norm=matplotlib.colors.Normalize(vmin=0, vmax=resized_attention.max()))
        axs[row, col].axis('off')
        axs[row, col].set_title(f'Step {idx + 1}\nWord: {current_word}')

    plt.tight_layout()
    plt.show()


def evaluate(model, test_set, vocabulary, mask_func, device, max_length, beam_width=5):
    """评估模型性能"""
    model.eval()
    metrics = np.zeros(4)
    p_bar = tqdm(range(len(test_set)), desc='evaluating')
    for i in p_bar:
        _, caption = test_set.get_pair(i)
        image, _, _ = test_set[i]
        caption_generated, score, _ = generate_by_beam_search(model, image.unsqueeze(0), vocabulary, device, max_length,
                                                              mask_func, beam_width)
        metrics += np.array([score,
                             bleu(caption_generated, caption, vocabulary),
                             rouge_l(caption_generated, caption, vocabulary),
                             meteor(caption_generated, caption, vocabulary)])
        value = metrics / (i + 1)
        p_bar.set_postfix(score=value[0], bleu=value[1], rouge_l=value[2], meteor=value[3])


if __name__ == "__main__":
    # =================== Example parameters for the model ===================
    img_size = 224  # Size of the input image
    in_channels = 3  # Number of input channels (for RGB images)
    patch_size = 16  # Size of each patch
    emb_size = 96  # Embedding size
    num_layers = 6  # Number of layers in both encoder and decoder
    num_heads = 8  # Number of attention headsOpenhimer
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
    vocabulary_path = 'vocabulary/vocab.json'
    word2vec_cache_path = 'vocabulary/word2vec.npy'
    dataset_root = 'data/deepfashion-multimodal'
    train_labels_path = 'data/deepfashion-multimodal/train_captions.json'
    test_labels_path = 'data/deepfashion-multimodal/test_captions.json'

    # =================== Vocabulary and Image transforms ===================
    vocabulary = Vocabulary(vocabulary_path)
    transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    mask_func = partial(create_masks, pad_idx=vocabulary.pad, num_heads=num_heads)

    # =================== Initialize the model ===================
    model = ImageCaptioningModel(img_size, in_channels, patch_size, emb_size, len(vocabulary), num_layers, num_heads,
                                 expansion, dropout,
                                 pretrained_embeddings=vocabulary.get_word2vec(cache_path=word2vec_cache_path))

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    # =================== Prepare for Training ===================

    train_set = ImageTextDataset(dataset_root,
                                 train_labels_path,
                                 vocabulary=vocabulary,
                                 max_seq_len=seq_length,
                                 transform=transform,
                                 max_cache_memory=32 * 1024 ** 3)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # =================== Start Training ===================
    # train(model, train_loader, criterion, optimizer, mask_func, save_path='models/model_transformer.pth',
    #       epochs=epochs, device=device, experiment_name=experiment_name)

    #  =================== Model Inference ===================
    test_set = ImageTextDataset(dataset_root,
                                test_labels_path,
                                vocabulary=vocabulary,
                                max_seq_len=seq_length,
                                transform=transform)

    image_path, caption = test_set.sample()
    original_image = Image.open(image_path)
    image = transform(original_image).unsqueeze(0)
    caption_generated, score, attn_images = generate_by_beam_search(model, image, vocabulary, device, seq_length,
                                                                    mask_func, beam_width=5)

    print('caption: ', caption, '\n\ngenerated: ', caption_generated, '\n')
    print('best score: ', score)  # 序列全概率的负对数似然, 越小越好
    print('bleu score: ', bleu(caption_generated, caption, vocabulary))
    print('rouge-l score: ', rouge_l(caption_generated, caption, vocabulary))
    print('meteor score: ', meteor(caption_generated, caption, vocabulary))

    visualize_attention(original_image, attn_images)  # 绘制注意力图

    evaluate(model, test_set, vocabulary, mask_func, device, seq_length, beam_width=5)  # 在测试集上评估模型
