import torch
from PIL import Image

from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess("clip_feature_extractor", model_type="ViT-B-16",
                                                                  is_eval=True, device=device)

im_path = '/Users/gillevi/Projects/ocr/data/icdar13/images/word_1.png'

raw_image = Image.open(im_path)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
for text in ['Tiredness', 'A sign saying Tiredness', 'text of Tiredness']:
    sample = {"image": image, "text_input": text}
    clip_features = model.extract_features(sample)

    image_features = clip_features.image_embeds_proj
    text_features = clip_features.text_embeds_proj

    sims = (image_features @ text_features.t())
    print(text, sims[0][0].item())
    gil = 1

