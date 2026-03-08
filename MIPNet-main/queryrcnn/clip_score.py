import torch
import torch.nn as nn

from .CLIP.clip import clip_feature_surgery

from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, InterpolationMode


img_resize = Compose([
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
])



# def get_clip_score_from_feature(model, image, text_features, temp=100.):
#     # size of image: [b, 3, 224, 224]
#     image = img_resize(image)
#     image_features = model.encode_image(image)
#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#
#     probs = temp * clip_feature_surgery(image_features, text_features)[:, 1:, :]
#     similarity = torch.mean(probs.softmax(dim=-1), dim=1, keepdim=False)
#
#     loss = 1. - similarity[:, 0]
#     loss = torch.sum(loss) / len(loss)
#     return loss


def get_clip_score_from_feature(model, image, text_features, temp=100.):
    """
    计算 CLIP 特征空间中的相似度损失。

    Args:
        model (nn.Module): CLIP 图像编码器模型。
        image (torch.Tensor): 增强后的图像，形状为 [B, 3, H, W]。
        text_features (torch.Tensor): 预定义的文本特征，形状为 [1, D]。
        temp (float): 温度参数，用于缩放损失。

    Returns:
        torch.Tensor: 标量相似度损失。
    """
    print("----- get_clip_score_from_feature -----")
    print(f"Input image shape: {image.shape}")  # 打印输入图像的形状

    # 假设 img_resize 是一个已定义的函数，用于调整图像大小
    print(f"image: {image}")
    image = img_resize(image)
    print(f"Resized image: {image}")  # 打印调整大小后的图像形状

    image_features = model.encode_image(image)
    print(f"Encoded image_features : {image_features}")  # 打印编码后的图像特征形状

    # 归一化图像特征
    # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # print(f"Normalized image_features: {image_features}")  # 打印归一化后的图像特征

    # 调用 clip_feature_surgery 函数

    probs = temp * clip_feature_surgery(image_features, text_features)[:, 1:, :]
    print(f"Probs shape after clip_feature_surgery and scaling: {probs.shape}")  # 打印 probs 的形状
    print(f"Probs tensor: {probs}")  # 打印 probs 的值

    # 计算相似度
    probs_softmax = probs.softmax(dim=-1)
    print(f"Probs after softmax: {probs_softmax}")  # 打印 softmax 后的 probs

    similarity = torch.mean(probs_softmax, dim=1, keepdim=False)
    print(f"Similarity shape: {similarity.shape}")  # 打印 similarity 的形状
    print(f"Similarity tensor: {similarity}")  # 打印 similarity 的值

    # 计算损失
    loss = 1. - similarity[:, 0]
    print(f"Loss before averaging: {loss}")  # 打印每个样本的损失值

    loss = torch.sum(loss) / len(loss)
    print(f"Final loss: {loss.item()}")  # 打印最终的损失值
    print("----- End of get_clip_score_from_feature -----\n")

    return loss




class L_clip_from_feature(nn.Module):
    def __init__(self, temp=100.):
        super(L_clip_from_feature, self).__init__()
        self.temp = temp
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, model, x, text_features):
        k1 = get_clip_score_from_feature(model, x, text_features, self.temp)
        return k1


def get_clip_score_MSE(res_model, pred, inp, weight):
    stack = img_resize(torch.cat([pred, inp], dim=1))
    pred_image_features = res_model.encode_image(stack[:, :3, :, :])
    inp_image_features = res_model.encode_image(stack[:, 3:, :, :])

    MSE_loss = 0
    for feature_index in range(len(weight)):
        MSE_loss = MSE_loss + weight[feature_index] * F.mse_loss(pred_image_features[1][feature_index], inp_image_features[1][feature_index])

    return MSE_loss


class L_clip_MSE(nn.Module):
    def __init__(self):
        super(L_clip_MSE, self).__init__()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, model, pred, inp, weight=None):
        if weight is None:
            weight = [1.0, 1.0, 1.0, 1.0, 0.5]
        res = get_clip_score_MSE(model, pred, inp, weight)
        return res
