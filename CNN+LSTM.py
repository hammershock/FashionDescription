from datetime import datetime, timedelta
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import torch.nn as nn

from datasets import ImageTextDataset
from vocabulary import Vocabulary


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_dim):
        """
        CNN图像编码器，得到图像整体特征
        :param out_dim: 输出特征维度
        :param in_channels: 图像通道数
        """
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        out_shape = (256, 8, 8)  # 根据实际输出尺寸进行调整
        self.fc = nn.Linear(in_features=out_shape[0] * out_shape[1] * out_shape[2], out_features=out_dim)

    def forward(self, x):
        x = self.conv_layers(x)  # (batch, 128, 8, 8)
        x = x.view(x.size(0), -1)  # Flatten, (batch, 128*8*8)
        x = self.fc(x)
        return x  # (batch, out_dim)


class ImageTextModel(nn.Module):
    """
    CNN做图像编码器，得到整体特征；LSTM做文本解码器，输出图像描述序列
    """

    def __init__(self, vocabulary_size, in_channels, image_embed_dim, text_embed_dim, hidden_size=512,
                 pretrained_embeddings=None):
        super(ImageTextModel, self).__init__()

        self.image_embed = ConvNet(in_channels, image_embed_dim)  # CNN整体表示

        if pretrained_embeddings is not None:
            assert pretrained_embeddings.shape == (vocabulary_size, text_embed_dim), "预训练嵌入向量尺寸不匹配"
            self.text_embed = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=False)
        else:
            self.text_embed = nn.Embedding(vocabulary_size, text_embed_dim)

        self.lstm = nn.LSTM(image_embed_dim + text_embed_dim, hidden_size, num_layers=2, batch_first=True)

        self.fc_out = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, images, seq, hidden_state: tuple = None, image_feats=None):
        """
        :param images: (batch, channels, img_size, img_size)
        :param seq: (batch, seq_len)
        :param hidden_state
        :param image_feats: 图像特征
        :return:
        """
        if image_feats is None:
            image_feats = self.image_embed(images)  # (batch, image_embed_dim)

        image_embed = image_feats.unsqueeze(1).repeat(1, seq.size(1), 1)  # (batch, seq_len, image_embed_dim)

        text_embed = self.text_embed(seq)  # (batch, seq_len, text_embed_dim)
        embeddings = torch.cat((image_embed, text_embed), dim=2)  # -> (batch, seq_len, text_embed_dim+image_embed_dim)

        lstm_output, hidden_state = self.lstm(embeddings, hidden_state)  # (batch, seq_len, hidden_size)

        output = self.fc_out(lstm_output)  # (batch, seq_len, vocabulary_size)
        return output, hidden_state, image_feats


def train(model, train_loader, criterion, optimizer, save_path, device, epochs=10,
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

            optimizer.zero_grad()
            prediction, _, _ = model(image, input_seq)  # (batch, seq_len - 1, vocabulary_size)
            _, _, vocab_size = prediction.shape
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


def predict(model, image, transform, vocab, device, max_length):
    model.to(device)
    image = transform(image).unsqueeze(0).to(device)  # -> (1, channel, img_size, img_size)
    seq = [vocab.start]

    h, feats = None, None
    for _ in range(max_length):
        with torch.no_grad():
            inp = torch.tensor([seq[-1]], dtype=torch.long, device=device).unsqueeze(0)
            output, h, feats = model(image, inp, h, feats)

        predicted = output.argmax(2)[:, -1]
        seq.append(predicted.item())
        if predicted.item() == vocab.end:
            break

    return vocab.decode(seq)


if __name__ == "__main__":
    # =========== Example parameters for the model ===========
    img_size = 256  # Size of the input image
    in_channels = 3  # Number of input channels (for RGB images)
    text_emb_size = 96  # Embedding size
    img_emb_size = 64
    hidden_size = 256
    dropout = 0.1  # Dropout rate
    lr = 5e-4  # Learning rate
    epochs = 500
    batch_size = 64  # Batch size
    seq_length = 128  # Max length of the caption sequence
    image_embed_dim = 64
    text_embed_dim = 96
    channels = 3
    save_path = 'models/model_transformer.pth'
    experiment_name = 'fashion_description_lstm'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ==================== Build Vocabulary ====================

    vocabulary = Vocabulary('vocab.json')

    # ==================== Define image transforms ====================

    transform = Compose([
        Resize((img_size, img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ==================== Initialize the model ====================
    model = ImageTextModel(len(vocabulary), in_channels, img_emb_size, text_emb_size, hidden_size,
                           pretrained_embeddings=vocabulary.get_word2vec()).to(device)

    model.load_state_dict(torch.load('models/model_lstm.pth'))

    # ==================== Prepare for training ====================
    train_set = ImageTextDataset('data/deepfashion-multimodal',
                                   'data/deepfashion-multimodal/train_captions.json',
                                   vocabulary=vocabulary,
                                   max_seq_len=seq_length,
                                   transform=transform,
                                   max_cache_memory=32 * 1024 ** 3)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # =================== Start Training ===================
    # train(model, train_loader, criterion, optimizer,
    #       save_path='models/model_lstm.pth',
    #       device=device,
    #       epochs=epochs,
    #       experiment_name='lstm')

    #  =================== Model Inference ===================
    image_path, caption = train_set.sample()

    caption_generated = predict(model, Image.open(image_path), transform, vocabulary, device, seq_length)
    print(caption, '\n\n', caption_generated)
