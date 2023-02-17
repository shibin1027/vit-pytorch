import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from vit_pytorch.vit import Transformer

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]  # num_patches=65, encoder_dim=1024

        self.to_patch = encoder.to_patch_embedding[0]  # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)'
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])  # nn.LayerNorm(patch_dim),nn.Linear(patch_dim, dim),nn.LayerNorm(dim),

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]  # patch_dim=3072

        # decoder parameters
        self.decoder_dim = decoder_dim  # 512
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))  # 512
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)  # 65, 512
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)  # 512,3072

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.to_patch(img)  # 8,3,256,256 -> 8,64,3072
        batch, num_patches, *_ = patches.shape  # num_patches=64

        # patch to encoder tokens and add positions
        
        tokens = self.patch_to_emb(patches)  # 8,64,1024
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]  # 8,64,1024

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)  # 64*0.75=48
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)  # 3,64
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:] # 3,48  3,16

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]  # 8,1
        tokens = tokens[batch_range, unmasked_indices]  # 8,16,1024

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]  # 8,48,3072

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)  # 8,16,1024

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)  # 8,16,512

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)  # 8,16,512

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked) # 512 -> 8,48,512
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices) # 8,48,512

        # concat the masked tokens to the decoder tokens and attend with decoder
        
        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device) # 8,64,512
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens) # 8,64,512

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]  # 8,48,512
        pred_pixel_values = self.to_pixels(mask_tokens)  # 8,48,3072

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)  # 8,48,3072  8,48,3072
        return recon_loss



if __name__ == '__main__':
    from vit_pytorch import ViT

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048
    )

    mae = MAE(
        encoder = v,
        masking_ratio = 0.75,   # the paper recommended 75% masked patches
        decoder_dim = 512,      # paper showed good results with just 512
        decoder_depth = 6       # anywhere from 1 to 8
    )

    images = torch.randn(8, 3, 256, 256)

    loss = mae(images)
    loss.backward()

    # that's all!
    # do the above in a for loop many times with a lot of images and your vision transformer will learn

    # save your improved vision transformer
    # torch.save(v.state_dict(), './trained-vit.pt')
    