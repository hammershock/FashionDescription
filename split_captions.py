"""
这个脚本用于将data中提供的captions的json文件打散为较小的短句，方便训练。
"""
import json
import os


def transform_caption_json(filename, root_path, output_path):
    with open(os.path.join(root_path, filename), 'rb') as fr:
        captions = json.load(fr)
        new_captions = []

        for image_file_name, caption_text in captions.items():
            new_captions.extend([(image_file_name, text_piece + '.') for text_piece in caption_text.split('.') if text_piece])

    with open(output_path, 'w') as fw:
        json.dump(new_captions, fw)


if __name__ == "__main__":
    root_dir = 'data/deepfashion-multimodal'

    filename = 'train_captions.json'
    transform_caption_json(filename, root_dir, 'data/deepfashion-multimodal/train_captions_split.json')
