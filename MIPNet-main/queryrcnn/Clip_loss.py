import torch
import torch.nn as nn
import torch.nn.functional as F
from .CLIP import clip


class CLIPLoss(nn.Module):
    def __init__(self, device, enhance_prompts, negative_prompts):
        super(CLIPLoss, self).__init__()
        self.device = device
        self.model, _ = clip.load("CS-ViT-B/32", device=device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.enhance_features = self._compute_prompt_features(enhance_prompts) if enhance_prompts else None
        self.negative_features = self._compute_prompt_features(negative_prompts,
                                                               aggregate=False) if negative_prompts else None

    def _compute_prompt_features(self, prompts, aggregate=True):

        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            features = F.normalize(features, dim=-1)

        return features.mean(dim=0, keepdim=True) if aggregate else features

    def forward(self, images):

        images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, -1, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, -1, 1, 1)
        images = (images - mean) / std

        images = images.to(self.device)

        image_features = self.model.encode_image(images)  # [B, dim]
        image_features = F.normalize(image_features, dim=-1)

        if self.enhance_features is not None:
            similarity_pos = (image_features @ self.enhance_features.T).squeeze(dim=-1)  # [B]

        if self.negative_features is not None:
            similarity_neg = (image_features @ self.negative_features.T)  # [B, K]

        exp_pos = torch.exp(similarity_pos)  # [B]

        exp_neg_sum = torch.exp(similarity_neg).sum(dim=-1)

        loss = -torch.log(exp_pos / (exp_pos + exp_neg_sum)).mean()

        return loss


class SemanticLoss(nn.Module):
    def __init__(self, enhance_prompts, negative_prompts):
        super(SemanticLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.clip_loss_fn = CLIPLoss(device=self.device, enhance_prompts=enhance_prompts,
                                     negative_prompts=negative_prompts)
    def forward(self, enhanced_image):

        enhance_loss = self.clip_loss_fn(enhanced_image)

        total_loss = enhance_loss

        return total_loss


