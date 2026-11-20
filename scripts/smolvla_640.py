"""SmolVLA image embedding for the project's 640×480 cameras."""

import torch.nn.functional as F
from einops import rearrange
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


def embed_640x480(self, image):
    vision = self.get_vlm_model().vision_model
    connector = self.get_vlm_model().connector
    if image.shape[-2:] != (480, 640):
        raise ValueError(f"Expected a 640×480 camera frame, got {tuple(image.shape)}")
    patches = vision(pixel_values=image.to(vision.dtype)).last_hidden_state
    height, width = (length // vision.patch_size for length in image.shape[-2:])
    grid = rearrange(patches, "b (h w) c -> b h w c", h=height, w=width)
    # The stock connector infers a square grid. Preserve the rectangular grid
    # and pad only its incomplete 4×4 patch groups before the same projection.
    size = connector.scale_factor
    grid = F.pad(grid, (0, 0, 0, -width % size, 0, -height % size))
    groups = rearrange(grid, "b (h i) (w j) c -> b (h w) (i j c)", i=size, j=size)
    return connector.modality_projection(groups)


def enable_640x480():
    SmolVLMWithExpertModel.embed_image = embed_640x480
