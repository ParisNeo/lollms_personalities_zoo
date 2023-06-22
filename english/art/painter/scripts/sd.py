from pathlib import Path
import os
import sys
from lollms.paths import LollmsPaths
import time
import numpy as np
import sys
import argparse
import torch
import importlib
from tqdm import tqdm


class SD:
    def __init__(self, lollms_path:LollmsPaths, gpt4art_config, wm = "Artbot"):
        # Get the current directory
        root_dir = lollms_path.personal_path
        current_dir = Path(__file__).resolve().parent

        # Store the path to the script
        shared_folder = root_dir/"shared"
        self.sd_folder = shared_folder / "sd"
        self.output_dir = root_dir / "outputs/sd"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        sys.path.append(str(self.sd_folder))
        self.text2image_module = self.get_text2image()

        # Add the sd folder to the import path
        
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--prompt",
            type=str,
            nargs="?",
            default="a painting of a virus monster playing guitar",
            help="the prompt to render"
        )
        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            help="dir to write results to",
            default=str(self.output_dir)
        )
        parser.add_argument(
            "--skip_grid",
            action='store_true',
            help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        )
        parser.add_argument(
            "--skip_save",
            action='store_true',
            help="do not save individual samples. For speed measurements.",
        )
        parser.add_argument(
            "--ddim_steps",
            type=int,
            default=50,
            help="number of ddim sampling steps",
        )
        parser.add_argument(
            "--plms",
            action='store_true',
            help="use plms sampling",
        )
        parser.add_argument(
            "--dpm_solver",
            action='store_true',
            help="use dpm_solver sampling",
        )
        parser.add_argument(
            "--laion400m",
            action='store_true',
            help="uses the LAION400M model",
        )
        parser.add_argument(
            "--fixed_code",
            action='store_true',
            help="if enabled, uses the same starting code across samples ",
        )
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=0.0,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=1,
            help="sample this often",
        )
        parser.add_argument(
            "--H",
            type=int,
            default=512,
            help="image height, in pixel space",
        )
        parser.add_argument(
            "--W",
            type=int,
            default=512,
            help="image width, in pixel space",
        )
        parser.add_argument(
            "--C",
            type=int,
            default=4,
            help="latent channels",
        )
        parser.add_argument(
            "--f",
            type=int,
            default=8,
            help="downsampling factor",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=1,
            help="how many samples to produce for each given prompt. A.k.a. batch size",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=0,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=7.5,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )
        parser.add_argument(
            "--from-file",
            type=str,
            help="if specified, load prompts from this file",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="configs/stable-diffusion/v1-inference.yaml",
            help="path to config which constructs model",
        )
        parser.add_argument(
            "--ckpt",
            type=str,
            default="models/ldm/stable-diffusion-v1/model.ckpt",
            help="path to checkpoint of model",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=-1,
            help="the seed (for reproducible sampling)",
        )
        parser.add_argument(
            "--precision",
            type=str,
            help="evaluate at this precision",
            choices=["full", "autocast"],
            default="autocast"
        )
        opt = parser.parse_args()

        if opt.laion400m:
            print("Falling back to LAION 400M model...")
            opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
            opt.ckpt = "models/ldm/text2img-large/model.ckpt"
            opt.outdir = "outputs/txt2img-samples-laion400m"
        else:
            opt.ckpt = root_dir/ "shared" / "sd_models"/ gpt4art_config["model_name"]

        opt.ddim_steps = gpt4art_config.get("ddim_steps",50)
        opt.scale = gpt4art_config.get("scale",7.5)
        opt.W = gpt4art_config.get("W",512)
        opt.H = gpt4art_config.get("H",512)
        opt.skip_grid = gpt4art_config.get("skip_grid",True)
        opt.batch_size = gpt4art_config.get("batch_size",1)
        opt.num_images = gpt4art_config.get("num_images",1)
        
        config = self.text2image_module.OmegaConf.load(f"{self.sd_folder / opt.config}")
        self.model = self.text2image_module.load_model_from_config(config, f"{opt.ckpt}")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(device)

        if gpt4art_config["sampler_name"].lower()=="dpms":
            self.sampler = self.text2image_module.DPMSolverSampler(self.model)
        elif gpt4art_config["sampler_name"].lower()=="plms":
            self.sampler = self.text2image_module.PLMSSampler(self.model)
        else:
            self.sampler = self.text2image_module.DDIMSampler(self.model)
        

        os.makedirs(opt.outdir, exist_ok=True)

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        
        self.wm_encoder = self.text2image_module.WatermarkEncoder()
        self.wm_encoder.set_watermark('bytes', wm.encode('utf-8'))


        self.opt = opt


    def get_text2image(self):
        text2img_script_path = self.sd_folder / "scripts/txt2img.py"
        if text2img_script_path.exists():
            module_name = text2img_script_path.stem  # Remove the ".py" extension
            # use importlib to load the module from the file path
            loader = importlib.machinery.SourceFileLoader(module_name, str(text2img_script_path))
            text2image_module = loader.load_module()
            return text2image_module

    def generate(self, prompt, num_images=1, seed = -1):
        self.opt.seed=seed
        self.opt.num_images=num_images
        outpath = self.opt.outdir
        batch_size = 1
        n_rows = self.opt.n_rows if self.opt.n_rows > 0 else batch_size
        self.text2image_module.seed_everything(self.opt.seed)

        if not self.opt.from_file:
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {self.opt.from_file}")
            with open(self.opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(self.text2image_module.chunk(data, batch_size))

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1

        start_code = None
        if self.opt.fixed_code:
            start_code = torch.randn([self.opt.n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f], device=device)

        precision_scope = self.text2image_module.autocast if self.opt.precision=="autocast" else self.text2image_module.nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in self.text2image_module.trange(self.opt.num_images, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if self.opt.scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f]
                            samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=self.opt.batch_size,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=self.opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=self.opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = self.text2image_module.check_safety(x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            if not self.opt.skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * self.text2image_module.rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = self.text2image_module.Image.fromarray(x_sample.astype(np.uint8))
                                    img = self.text2image_module.put_watermark(img, self.wm_encoder)
                                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1

                            if not self.opt.skip_grid:
                                all_samples.append(x_checked_image_torch)

                    if not self.opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = self.text2image_module.rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = self.text2image_module.make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * self.text2image_module.rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = self.text2image_module.Image.fromarray(grid.astype(np.uint8))
                        img = self.text2image_module.put_watermark(img, self.wm_encoder)
                        img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        grid_count += 1

                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"+f" \nEnjoy.")
        
        files =[f for f in (self.output_dir/"samples").iterdir()]
        return files[-num_images:]
        

   