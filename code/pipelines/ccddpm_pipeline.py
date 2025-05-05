from typing import List, Optional, Union, Tuple
import torch
from diffusers import DDPMPipeline, ImagePipelineOutput, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor


class CCDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler):
        if not isinstance(unet, UNet2DConditionModel):
            raise ValueError(
                "CCDDPMPipeline requires a UNet2DConditionModel for class conditioning."
            )
        super().__init__(unet, scheduler)

    # overwrite the __call__method to accept class labels and encoder hidden states.
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        *,
        class_labels: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Determine shape for initial noise
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                *self.unet.config.sample_size,
            )

        # Sample gaussian noise to begin loop
        if self.device.type == "mps":
            image = randn_tensor(
                image_shape, generator=generator, dtype=self.unet.dtype
            )
            image = image.to(self.device)
        else:
            image = randn_tensor(
                image_shape,
                generator=generator,
                device=self.device,
                dtype=self.unet.dtype,
            )

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising loop
        for t in self.progress_bar(self.scheduler.timesteps):
            model_output = self.unet(
                image,
                t,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
            ).sample
            image = self.scheduler.step(
                model_output, t, image, generator=generator
            ).prev_sample

        # Post-process to image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
