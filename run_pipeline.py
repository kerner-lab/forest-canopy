"""
Running fine-tuning pipeline for predicting
GEDI using embeddings with pre-trained models
"""
from terratorch.registry import BACKBONE_REGISTRY
import timm
import argparse
from models import *
from models import prithvi


def get_args():
    parser = argparse.ArgumentParser(description='Running finetuning on GFMs')
    parser.add_argument('--root_dir', help='Root directory for saving things')
    parser.add_argument('--data_dir', help='Data directory of dataset')
    parser.add_argument('--model', help='Model to test')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    data_dir = args.data_dir
    root_dir = args.root_dir
    model = args.model

    # embedding dictionary, key-pairs are embedding name and size
    embedding_dict = {
        'ae': 64
    }

    # Things will be ugly at first and I will move things into their own modules etc later
    def get_model(model_name, embedding):
        """
        :param model_name: name of model to finetune
        :param embedding: embedding to use to finetune
        :return:
        """
        if model_name == 'prithvi':
            backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_tiny_tl", pretrained=True)
            return prithvi.PrithviRegressor(backbone, img_channels=embedding_dict[embedding], num_outputs=1)
        else:
            return None



