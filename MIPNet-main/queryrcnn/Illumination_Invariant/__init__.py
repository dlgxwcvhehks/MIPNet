from .main import NAFNet

def build_net(
    in_channels,
    weight_path=r'E:\code\11object\FeatEnHancer-main\FeatEnHancer-main\low-light-object-detection-detectron2\queryrcnn\featenhancer\depth_anything_v2\depth_anything_v2_vits.pth',
    enc_blk_nums=[1, 1, 1, 1],
    dec_blk_nums=[1, 1, 1, 1]
):

    model = NAFNet(
        img_channel=in_channels,
        weight_path=weight_path,
        enc_blk_nums=enc_blk_nums,
        dec_blk_nums=dec_blk_nums
    )
    return model
