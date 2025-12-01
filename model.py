import torch
import torch.nn as nn
import timm

class BiomassSwinModel(nn.Module):
    """
    使用 Swin Transformer 进行多输出回归
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', pretrained=False, num_outputs=5, dropout=0.3):
        super().__init__()
        # timm 载入 backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # num_classes=0表示去掉原分类头
        in_features = self.backbone.num_features  # backbone 输出特征维度

        # 回归头
        self.head = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features//2, num_outputs)
        )

    def forward(self, x):
        x = self.backbone(x)  # [B, C]
        out = self.head(x)    # [B, num_outputs]
        return out

if __name__ == '__main__':
    model = BiomassSwinModel().cuda()
    state_dict = torch.load('D:/Competition/models/swin_base_patch4_window7_224_22kto1k.pth', map_location='cpu')
    model.load_state_dict(state_dict, strict=False)