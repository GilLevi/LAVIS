import torch
from PIL import Image
from pathlib import Path
import json
import pickle
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import torch.nn.functional as F


ANN_BASE_PATH = '/Users/gillevi/Projects/lavis/data/flickr30k/annotations'
BASE_IM_PATH = '/Users/gillevi/Projects/lavis/data/flickr30k/images'
BASE_OUT_PATH = '/Users/gillevi/Projects/lavis/data/flickr30k/clip_features'


def load_annotations(split='test'):
    with open(f'{ANN_BASE_PATH}/{split}.json') as f:
        anns = json.load(f)

    return anns


def extract_features(split='test', model_type='ViT-B-32', redo=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess("clip_feature_extractor", model_type=model_type,
                                                                      is_eval=True, device=device)

    base_out_path = Path(BASE_OUT_PATH, model_type, split)
    base_out_path.mkdir(parents=True, exist_ok=True)
    with open(f'{ANN_BASE_PATH}/{split}.json') as f:
        anns = json.load(f)

    for ann in tqdm(anns):
        out_file = Path(base_out_path, f'{ann["image"].split("/")[-1]}.pickle')
        if out_file.exists() and not redo:
            continue

        cur_features_dict = {}
        im_path = Path(BASE_IM_PATH, ann['image'])
        raw_image = Image.open(im_path)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        sample = {"image": image}
        im_features = model.extract_features(sample)
        im_features = F.normalize(im_features, dim=-1)

        cur_features_dict['im_features'] = im_features.detach().cpu().numpy()
        cur_features_dict['image_name'] = ann['image'].split('/')[-1]

        for ind, text_t in enumerate(ann['caption']):
            text = txt_processors['eval'](text_t)
            if text != text_t:
                gil = 1
            sample = {"text_input": text}
            text_features = model.extract_features(sample)
            text_features = F.normalize(text_features, dim=-1)
            cur_features_dict[f'text_features_{ind}'] = text_features.detach().cpu().numpy()
            similarity = (im_features @ text_features.t())[0][0].item()
            cur_features_dict[f'similarity_{ind}'] = similarity

        with open(out_file, 'wb+') as f:
            pickle.dump(cur_features_dict, f)


if __name__ == '__main__':
    extract_features()
