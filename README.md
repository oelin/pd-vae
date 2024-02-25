# VL-VAE

VL-VAE is a transformer-based VAE architecture that supports [progressive decoding](https://www.youtube.com/watch?v=UphN1_7nP8U) through variable-length latent embeddings.


## Examples

Progressive decoding examples from CelebA-HQ-256x256. 

https://github.com/oelin/pd-vae/assets/42823429/39b45934-3900-4987-9888-b6412b403f6a

https://github.com/oelin/pd-vae/assets/42823429/0f6c8bea-2001-4627-b618-a473d74dab16

https://github.com/oelin/pd-vae/assets/42823429/235fea18-2fde-4692-9076-2575f1b28db5

https://github.com/oelin/pd-vae/assets/42823429/47f44778-c918-40fe-9e5d-f919ce4c63f2


## Architecture

VL-VAE uses a straightforward architecture consisting of two headless transformers that implement the encoder and decoder networks respectively. Unlike conventional autoencoders, the architecture *does not neccessarily* include downsampling layers. Instead, compression is enforced by randomly truncating the encoder's output (i.e. latent embeddings) during training. We sample truncation lengths according to an exponential distribution.


## TODO

- [ ] Experiment with alternative attention mechanisms (NAT, axial, etc).
- [ ] Experiment with alternative positional embedding methods.
- [ ] Experiment with alternative patch embeddings.
- [ ] Scale up to 1024x1024 resolution.
