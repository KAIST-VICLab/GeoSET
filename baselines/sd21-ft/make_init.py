"""Write the stage-1 initialisation: the SD 2.1-base UNet with an 8-channel conv_in.

usage: python make_init.py <out.safetensors>
conv_in is widened from 4 to 8 input channels by duplicating its kernel and halving it
(SAR latent | noisy EO latent); conv_out keeps its 4 channels.
"""
import sys

from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

BASE = 'Manojb/stable-diffusion-2-1-base'
REVISION = '0094d483a120f3f33dafbd187ea4aa60d10de75c'


def main():
    if len(sys.argv) != 2:
        sys.exit(f'usage: python {sys.argv[0]} <out.safetensors>')
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder='unet', revision=REVISION)
    state = unet.state_dict()
    state['conv_in.weight'] = state['conv_in.weight'].repeat((1, 2, 1, 1)) * 0.5
    save_file(state, sys.argv[1])


if __name__ == '__main__':
    main()
