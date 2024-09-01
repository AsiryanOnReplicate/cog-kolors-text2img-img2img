# Prediction interface for Cog
from cog import BasePredictor, Input, Path
import os
import time
import torch
import shutil
from typing import List, Optional
from diffusers.utils import load_image
from diffusers import KolorsPipeline, KolorsImg2ImgPipeline


class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        print("Loading Kolors txt2img pipeline...")
        # self.txt2img_pipe = KolorsPipeline.from_pretrained(
        #     "Kwai-Kolors/Kolors-diffusers", 
        #     torch_dtype=torch.float16, 
        #     variant="fp16"
        # )
        self.txt2img_pipe = KolorsPipeline.from_pretrained(
            "./kolors-cache",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.txt2img_pipe.to("cuda")

        print("Loading Kolors img2img pipeline...")
        self.img2img_pipe = KolorsImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.txt2img_pipe.to("cuda")

        print("setup took: ", time.time() - start)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Expressive illustration of a pug puppy on a Water Slide basked in summer heat"
        ),
        negative_prompt: str = Input(
            description="Negative Input prompt",
            default=""
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=1024
        ),
        height: int = Input(
            description="Height of output image",
            default=1024
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=5
        ),
        strength: float = Input(
            description="Prompt strength when using img2img 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        kolors_kwargs = {}

        print(f"Prompt: {prompt}")
        if image:
            print("img2img mode")
            kolors_kwargs["image"] = self.load_image(image)
            kolors_kwargs["strength"] = strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            kolors_kwargs["width"] = width
            kolors_kwargs["height"] = height
            pipe = self.txt2img_pipe

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = pipe(**common_args, **kolors_kwargs)        
        output_paths = []

        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths