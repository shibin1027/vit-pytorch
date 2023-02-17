import math
import torch
from torch import nn

from vit_pytorch.vit import Transformer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

# main class

class T2TViT(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth = None, heads = None, mlp_dim = None, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., transformer = None, t2t_layers = ((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        layer_dim = channels  # 3
        output_image_size = image_size  # 224

        # t2t_layers = ((7, 4), (3, 2), (3, 2))
        for i, (kernel_size, stride) in enumerate(t2t_layers):
            # i=0 kernel_size=7 stride=4
            layer_dim *= kernel_size ** 2  # 3*49
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)  # (224-7+6)/4+1=56

            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2), # 1,3,244,244 -> 1,56*56,3*7*7
                Rearrange('b c n -> b n c'),  # 1,56*56,3*7*7
                Transformer(dim = layer_dim, heads = 1, depth = 1, dim_head = layer_dim, mlp_dim = layer_dim, dropout = dropout) if not is_last else nn.Identity(),
            ])  # 1,56*56,3*7*7
            # 1,(56,56),3*7*7 -> 1,3*7*7,56,56 -> 1,28*28,(3*7*7)*3*3 -> 1,28,28,3*3*3

        layers.append(nn.Linear(layer_dim, dim)) # 1,14*14,3*3*3 -> 1,196,512
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        print(x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n+1]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



if __name__ == '__main__':
    v = T2TViT(
        dim = 512,
        image_size = 224,
        depth = 5,
        heads = 8,
        mlp_dim = 512,
        num_classes = 1000,
        t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    )

    img = torch.randn(1, 3, 224, 224)

    preds = v(img) # (1, 1000)
    print(preds.shape)
    