import torch
import torch.nn.functional as F
from torch import nn
from vit_pytorch.vit import ViT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.efficient import ViT as EfficientViT

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

# classes

class DistillMixin:
    def forward(self, img, distill_token = None):
        # img: 2,3,256,256
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)  # 2,64,1024
        b, n, _ = x.shape  # b=2, n=64

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) # 1,1,1024 -> 2,1,1024
        x = torch.cat((cls_tokens, x), dim = 1)  # 2,65,1024
        x += self.pos_embedding[:, :(n + 1)]  # 2,65,1024

        if distilling:
            distill_tokens = repeat(distill_token, '() n d -> b n d', b = b)  # 1,1,1024 -> 2,1,1024
            x = torch.cat((x, distill_tokens), dim = 1)  # 2,66,1024

        x = self._attend(x)  # 2,66,1024

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]  # 2,65,1024  2,1024

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]  # 2,1024

        x = self.to_latent(x)
        out = self.mlp_head(x)  # 2,1000

        if distilling:
            return out, distill_tokens  # 2,1000  2,1024

        return out

class DistillableViT(DistillMixin, ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableT2TViT(DistillMixin, T2TViT):
    def __init__(self, *args, **kwargs):
        super(DistillableT2TViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = T2TViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableEfficientViT(DistillMixin, EfficientViT):
    def __init__(self, *args, **kwargs):
        super(DistillableEfficientViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = EfficientViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        return self.transformer(x)

# knowledge distillation wrapper

class DistillWrapper(nn.Module):
    def __init__(
        self,
        *,
        teacher,
        student,
        temperature = 1.,
        alpha = 0.5,
        hard = False
    ):
        super().__init__()
        assert (isinstance(student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT))) , 'student must be a vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, temperature = None, alpha = None, **kwargs):
        b, *_ = img.shape  # b=2
        alpha = alpha if exists(alpha) else self.alpha
        T = temperature if exists(temperature) else self.temperature

        with torch.no_grad():
            teacher_logits = self.teacher(img) # 2,1000

        student_logits, distill_tokens = self.student(img, distill_token = self.distillation_token, **kwargs) # 2,1000  2,1024
        distill_logits = self.distill_mlp(distill_tokens) # 1,1000

        loss = F.cross_entropy(student_logits, labels)

        if not self.hard:
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim = -1),
                F.softmax(teacher_logits / T, dim = -1).detach(),
            reduction = 'batchmean')  # calculate KL divergence for soft setting
            distill_loss *= T ** 2

        else:
            teacher_labels = teacher_logits.argmax(dim = -1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        return loss * (1 - alpha) + distill_loss * alpha



if __name__ == '__main__':

    from torchvision.models import resnet50

    teacher = resnet50(pretrained = True)

    v = DistillableViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    distiller = DistillWrapper(
        student = v,
        teacher = teacher,
        temperature = 3,           # temperature of distillation
        alpha = 0.5,               # trade between main loss and distillation loss
        hard = False               # whether to use soft or hard distillation
    )

    img = torch.randn(2, 3, 256, 256)
    labels = torch.randint(0, 1000, (2,))

    loss = distiller(img, labels)
    loss.backward()

    # after lots of training above ...

    pred = v(img) # (2, 1000)
    