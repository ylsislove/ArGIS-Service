import  os
import  torch
from    torch import nn
from    torchvision import models


def load_model(model_path, device):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 6)

    # 预训练模型存在 加载预训练模型参数
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("模型不存在，请检查！")
        exit(1)

    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model('model_dir/20210312/model_03111848_180.pth', device)
    print(model)
