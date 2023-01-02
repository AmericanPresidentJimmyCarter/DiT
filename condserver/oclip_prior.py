import torch
import open_clip
from dalle2_pytorch.train_configs import DiffusionPriorConfig, TrainDiffusionPriorConfig

# DEVICE = 'cuda:7'
DEVICE = 'cuda'
JSON_CONFIG_PATH = '/home/user/storage/fsx/hlky/h-14-prior/h-14-prior-checkpoint-official/prior_config.json'
PRIOR_PATH = '/home/user/storage/fsx/hlky/h-14-prior/h-14-prior-checkpoint-official/latest_model.pth'

tokenizer = open_clip.get_tokenizer("ViT-H-14")

def make_prior(
    prior_config: DiffusionPriorConfig, checkpoint_path: str, device: str = DEVICE
):
    # create model from config
    diffusion_prior = prior_config.create()
    model_training = torch.load(checkpoint_path, map_location="cpu")
    state_dict = model_training['model']
    diffusion_prior.load_state_dict(state_dict)
    diffusion_prior.eval()
    diffusion_prior.to(device)

    if device == "cpu":
        diffusion_prior.float()
    return diffusion_prior

# load entire config
train_config = TrainDiffusionPriorConfig.from_json_path(JSON_CONFIG_PATH)
prior_config = train_config.prior

# load model
prior = make_prior(prior_config=prior_config, checkpoint_path=PRIOR_PATH,
    device=DEVICE)
clip = prior.clip

def image_embeddings_for_text(captions):
    tokens = tokenizer(captions).to(DEVICE)
    output = prior.sample(tokens, num_samples_per_batch=5)
    return output
