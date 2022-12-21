import os
import base64
import io
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb
from torch import nn, optim
import torchvision
from tqdm import tqdm
import time
import numpy as np
from models2 import DiT_S_2_CA # DiT_XL_2_CA
import requests

from ema import ModelEma
from accelerate import Accelerator
from torch.autograd import Variable
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion

import ujson

accelerator = Accelerator()
device = accelerator.device

URL_BATCH = 'http://127.0.0.1:4456/batch'
URL_CONDITIONING = 'http://127.0.0.1:4455/conditioning'


def b64_string_to_tensor(s: str, device) -> torch.Tensor:
    tens_bytes = base64.b64decode(s)
    buff = io.BytesIO(tens_bytes)
    buff.seek(0)
    return torch.load(buff, device)


def decode_latents(vae, latents):
    print('latents', latents.size())
    image = None
    with torch.no_grad():
        image = vae.decode(latents / 0.18215).sample
        print('out', image.size())
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    return image


def sample(model, vae, diff_module, conds, unconds, sz, _device):
    z = torch.randn(conds.size()[0], 4, sz // 8, sz // 8, device=_device)
    z = torch.cat([z, z], 0)
    # print('conds', conds.size())
    # print('unconds', unconds.size())
    interleaved = torch.empty((conds.size()[0] * 2, conds.size()[1],
        conds.size()[2])).to(_device)
    for i in range(conds.size()[0] * 2):
        if i % 2 == 0:
            interleaved[i] = conds[i // 2]
        else:
            interleaved[i] = unconds[i // 2]
    model_kwargs = dict(conditioning=interleaved, cfg_scale=4.)

    # Sample images:
    samples = diff_module.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=_device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    return decode_latents(vae, samples)


def maybe_unwrap_model(model):
    if getattr(model, 'module', None) is not None:
        return model.module
    return model


def train(args):
    if os.path.exists(f"models/{args.run_name}/pytorch_model.bin"):
        resume = True
    else:
        resume = False
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
        accelerator.print(f"Starting run '{args.run_name}'....")
        accelerator.print(f"Batch Size check: {args.n_nodes * args.batch_size * args.accum_grad * len(args.devices)}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

    # wait for everyone to load vae
    accelerator.wait_for_everyone()

    model = DiT_S_2_CA(input_size=args.image_size // 8).to(device)
    # wait for everyone to load model
    accelerator.wait_for_everyone()

    accelerator.print(
        f"Number of Parameters: {sum([p.numel() for p in model.parameters()])}"
    )

    lr = 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    diff_module = create_diffusion(str(args.timesteps))

    if accelerator.is_main_process:
        wandb.watch(model)
        os.makedirs(f"results/{args.run_name}", exist_ok=True)
        os.makedirs(f"models/{args.run_name}", exist_ok=True)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        total_steps=args.total_steps,
        max_lr=lr,
        pct_start=0.05, # 0.1 if not args.finetune else 0.0,
        div_factor=25,
        final_div_factor=1 / 25,
        anneal_strategy="linear",
    )

    dataset = None
    model, optimizer, dataset, scheduler = accelerator.prepare(model, optimizer,
        dataset, scheduler)

    if resume:
        losses = []
        # accuracies = []
        start_step, total_loss = 0, 0
        accelerator.print("Loading last checkpoint....")
        accelerator.load_state(f"models/{args.run_name}/")

        # Fun hack to init weights
        #
        # unwrapped_model = accelerator.unwrap_model(model)
        # loaded = torch.load(f"models/{args.run_name}/pytorch_model.bin", map_location='cpu')
        # unwrapped_model.load_state_dict(loaded)
    else:
        losses = []
        # accuracies = []
        start_step, total_loss = 0, 0

    model_ema = None
    if args.ema:
        model_ema = ModelEma(
            model=model,
            decay=args.ema_decay,
            device='cpu',
            ema_model_path=args.ema_model_path,
        )

    accelerator.wait_for_everyone()

    pbar = tqdm(
        total=args.total_steps,
        initial=start_step,
    ) if accelerator.is_main_process else None

    model.train()
    step = 0
    epoch = 0

    # Do less on last GPU.
    # if accelerator.is_last_process:
    #     args.batch_size = args.batch_size // 3

    while step < args.total_steps:
        resp_dict = None
        try:
            resp = requests.post(url=URL_BATCH, json={'is_main': accelerator.is_last_process}, timeout=600)
            # resp_dict = resp.json()
            resp.encoding = 'UTF-8'
            resp_dict = ujson.loads(resp.text)
        except Exception:
            import traceback
            traceback.print_exc()
            continue

        if 'images' not in resp_dict or resp_dict['images'] is None or \
            'captions' not in resp_dict or resp_dict['captions'] is None or \
            'conditioning_flat' not in resp_dict or resp_dict['conditioning_flat'] is None or \
            'conditioning_full' not in resp_dict or resp_dict['conditioning_full'] is None or \
            'unconditioning_flat' not in resp_dict or resp_dict['unconditioning_flat'] is None or \
            'unconditioning_full' not in resp_dict or resp_dict['unconditioning_full'] is None:
            continue

        images = b64_string_to_tensor(resp_dict['images'], 'cpu')
        if images is None:
            continue
        images = images.tile((2,1,1,1))[0:args.batch_size].to(device)

        captions = resp_dict['captions']

        text_embeddings_full = b64_string_to_tensor(resp_dict['conditioning_full'],
            'cpu')
        if text_embeddings_full is None:
            continue
        text_embeddings_full = text_embeddings_full.tile((2,1,1))[0:args.batch_size].to(device)

        text_embeddings_full_uncond = b64_string_to_tensor(resp_dict['unconditioning_full'],
            'cpu')
        if text_embeddings_full_uncond is None:
            continue
        text_embeddings_full_uncond = text_embeddings_full_uncond.tile((2,1,1))[0:args.batch_size].to(device)

        if text_embeddings_full is None or text_embeddings_full_uncond is None:
            continue

        image_latents = vae.encode(images).latent_dist.sample() *  0.18215

        # text_full_for_step = text_embeddings_full
        # if (
        #     np.random.rand() < 0.1
        # ):  # 10% of the times -> unconditional training for classifier-free-guidance
        #     text_full_for_step = text_embeddings_full_uncond
        total_loss = 0
        for nts in range(args.timesteps):
            pred = diff_module.training_losses(
                getattr(maybe_unwrap_model(model), 'forward'),
                image_latents,
                torch.randint(0, args.timesteps, (image_latents.size()[0],),
                    device=device),
                model_kwargs={ 'conditioning': text_embeddings_full })
            loss = torch.mean(pred['loss'])
            # print(loss, pred)

            accelerator.backward(loss, retain_graph=nts != args.timesteps - 1)
            total_loss += loss.detach().cpu().numpy()
            # grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_norm).item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # acc = (pred.argmax(1) == image_indices).float()
        # acc = acc.mean()

        # del image_indices_cloned

        if accelerator.is_main_process:
            log = {
                "loss": total_loss / (step + 1),
                # "acc": total_acc / (step + 1),
                "curr_loss": loss.item(),
                # "curr_acc": acc.item(),
                "ppx": np.exp(total_loss / (step + 1)),
                "lr": optimizer.param_groups[0]["lr"],
            }
            pbar.set_postfix(log)
            wandb.log(log)

        if (
            model_ema is not None
            and accelerator.is_main_process
            and step % args.ema_update_steps == 0
            and step != 0
        ):
            accelerator.print(f"EMA weights are being updated and saved ({step=})")
            model_ema.update(maybe_unwrap_model(model))
            torch.save(maybe_unwrap_model(model), args.ema_model_path)

        # All of this is only done on the main process
        if step % args.log_period == 0 and accelerator.is_main_process:
            accelerator.print(
                f"Step {step} - loss {total_loss / (step + 1)} - ppx {np.exp(total_loss / (step + 1))}"
            )

            losses.append(total_loss / (step + 1))
            # accuracies.append(total_acc / (step + 1))

            model.eval()
            with torch.no_grad():
                n = 1
                images = images[: args.comparison_samples]
                image_latents = image_latents[: args.comparison_samples]
                captions = captions[: args.comparison_samples]
                text_embeddings_full = text_embeddings_full[: args.comparison_samples]
                samples = sample(maybe_unwrap_model(model), vae, diff_module,
                    text_embeddings_full, text_embeddings_full_uncond,
                    args.image_size, device)
                print('samples')
                    
                # recon_images = vae.decode(samples / 0.18215).sample
                # print('samples', samples.size())
                recon_images = decode_latents(vae, image_latents)

                if args.log_captions:
                    # cool_captions_data = torch.load("cool_captions.pth")
                    # cool_captions_text = cool_captions_data["captions"]
                    cool_captions_text = args.cool_captions_text

                    resp_dict = None
                    try:
                        resp = requests.post(url='http://127.0.0.1:4455/conditionings', json={
                            'captions': cool_captions_text,
                        }, timeout=600)
                        resp_dict = resp.json()
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        continue

                    cool_captions_embeddings_full = b64_string_to_tensor(resp_dict['full'],
                        device)

                    # cool_captions_embeddings = generate_clip_embeddings(clip_model,
                    #     text_tokens)

                    cool_captions = DataLoader(
                        TensorDataset(
                            cool_captions_embeddings_full.repeat_interleave(n, dim=0)
                        ),
                        batch_size=len(args.cool_captions_text),
                    )
                    cool_captions_sampled = []
                    st = time.time()
                    for caption_embedding in cool_captions:
                        caption_embedding = caption_embedding[0].float().to(device)
                        images_cool = sample(maybe_unwrap_model(model), vae,
                            diff_module, caption_embedding, text_embeddings_full_uncond,
                            args.image_size, device)

                        for s in images_cool:
                            cool_captions_sampled.append(s)

                    accelerator.print(
                        f"Took {time.time() - st} seconds to sample {len(cool_captions_text) * 2} captions."
                    )

                    # torchvision.utils.save_image(
                    #     torchvision.utils.make_grid(torch.Tensor(cool_captions_sampled), nrow=11),
                    #     os.path.join(
                    #         f"results/{args.run_name}", f"cool_captions_{step:03d}.png"
                    #     ),
                    # )

                    # cool_captions_sampled_ema = torch.stack(cool_captions_sampled_ema)
                    # torchvision.utils.save_image(
                    #     torchvision.utils.make_grid(cool_captions_sampled_ema, nrow=11),
                    #     os.path.join(f"results/{args.run_name}", f"cool_captions_{step:03d}_ema.png")
                    # )

                # log_images = samples

            # if accelerator.is_main_process:
            #     accelerator.save_state(f"models/{args.run_name}/")

            model.train()

            # torchvision.utils.save_image(
            #     torch.Tensor(log_images), os.path.join(f"results/{args.run_name}", f"{step:03d}.png"),
            # )

            log_data = [
                [captions[i]]
                + [wandb.Image(samples[i])]
                + [wandb.Image(images[i])]
                + [wandb.Image(recon_images[i])]
                for i in range(len(captions))
            ]
            log_table = wandb.Table(
                data=log_data, columns=["Caption", "Image", "Orig", "Recon"]
            )
            wandb.log({"Log": log_table})

            if args.log_captions:
                log_data_cool = [
                    [cool_captions_text[i]] + [wandb.Image(cool_captions_sampled[i])]
                    for i in range(len(cool_captions_text))
                ]
                log_table_cool = wandb.Table(
                    data=log_data_cool, columns=["Caption", "Image"]
                )
                wandb.log({"Log Cool": log_table_cool})
                del log_data_cool

            del samples, log_data

            if step % args.extra_ckpt == 0 and step != 0:
                # torch.save(
                #     model.state_dict(), f"models/{args.run_name}/model_{step}.pt"
                # )
                # torch.save(
                #     optimizer.state_dict(),
                #     f"models/{args.run_name}/model_{step}_optim.pt",
                # )
                accelerator.save_state(f"models/{args.run_name}/{step}/")

            # torch.save(model.state_dict(), f"models/{args.run_name}/model.pt")
            # torch.save(optimizer.state_dict(), f"models/{args.run_name}/optim.pt")
            # torch.save(
            #     {"step": step, "losses": losses, "accuracies": accuracies},
            #     f"results/{args.run_name}/log.pt",
            # )

        if accelerator.is_main_process and step % args.write_every_step == 0 and step != 0:
            accelerator.save_state(f"models/{args.run_name}/")

        if accelerator.is_main_process:
            # This is the main process only, so increment by the number of
            # devices.
            pbar.update(1)
            step += 1

        # del out_flat
        del images, pred, loss, image_latents
        del text_embeddings_full, text_embeddings_full_uncond

    accelerator.print(f"Training complete (steps: {step}, epochs: {epoch})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "ditc"
    args.model = "DiTC"
    args.dataset_type = "webdataset"
    args.total_steps = 2_000_000
    # Be sure to sync with TARGET_SIZE in utils.py and condserver/data.py
    args.batch_size = 8 # 112
    args.image_size = 256
    args.log_period = 75
    args.extra_ckpt = 10_000
    args.write_every_step = 25
    args.ema = False
    args.ema_decay = 0.9999
    args.ema_update_steps = 50_000
    args.ema_model_path = "ema_weights.ckpt"
    args.accum_grad = 1.
    args.num_codebook_vectors = 8192
    args.log_captions = True
    args.finetune = False
    args.comparison_samples = 2 # 12
    args.cool_captions_text = [
        "a cat is sleeping",
        "a painting of a clown",
        # "a horse",
        # "a river bank at sunset",
        # "bon jovi playing a sold out show in egypt. you can see the great pyramids in the background",
        # "The citizens of Rome rebel against the patricians, believing them to be hoarding all of the food and leaving the rest of the city to starve",
        # "King Henry rouses his small, weak, and ill troops, telling them that the less men there are, the more honour they will all receive.",
        # "Upon its outward marges under the westward mountains Mordor was a dying land, but it was not yet dead. And here things still grew, harsh, twisted, bitter, struggling for life.",
        # "the power rangers high five a dolphin",
        # "joe biden is surfing a giant wave while holding an ice cream cone",
    ]
    parallel_init_dir = "/data"
    args.parallel_init_file = f"file://{parallel_init_dir}/dist_file"
    args.wandb_project = "ditc-test"
    args.wandb_entity = "mbabbins"
    # args.cache_dir = "/data/cache"  # cache_dir for models
    # args.cache_dir = "/fsx/hlky/.cache"
    args.cache_dir = "/home/user/.cache"
    args.offload = False
    args.n_nodes = 1
    args.devices = [0,1,2,3,4,5,6,7]
    args.timesteps = 250

    # Testing:
    # args.dataset_path = '/home/user/Programs/Paella/models/6.tar'
    # args.dataset_path = "gigant/oldbookillustrations_2"
    args.dataset_path = "laion/laion-coco"
    accelerator.print("Launching with args: ", args)
    train(args)

