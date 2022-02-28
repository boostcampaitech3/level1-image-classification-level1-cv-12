import argparse
import os
from importlib import import_module

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_ensemble import select_model

from dataset_ensemble import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
#     model_cls = getattr(import_module("model"), args.model)
#     model = model_cls(
#         num_classes=num_classes
#     )
    model = select_model('efficientnet', 18)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, output_dir, args):
    """
    model_dir 부분만 모델이 위치한 경로 입력해서 사용하면 됩니다.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_dir = ['model/ensem','model/ensem2','model/ensem3','model/ensem4','model/ensem5']
    num_classes = MaskBaseDataset.num_classes  # 18
    oof_pred = None
    
    for directory in model_dir:
        model = load_model(directory, num_classes, device).to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'images')
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                preds.extend(pred.cpu().numpy())
        fold_pred = np.array(preds)
        if oof_pred is None:
            oof_pred = fold_pred / len(model_dir)
        else:
            oof_pred += fold_pred / len(model_dir)
        
    info['ans'] = np.argmax(oof_pred, axis=1)
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224, 224), help='resize size for image when you trained (default: (96, 128))')
#     parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/cv_output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, output_dir, args)