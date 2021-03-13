# coding=utf-8
# Created by Wang yu at 2021/3/12
import  io
import  sys
import  torch
from    torchvision import transforms
from    PIL import Image

sys.path.append("model_dir/20210312")
from    load import load_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model("model_dir/20210312/model_03111848_180.pth", device)
model.to(device)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((320, 480)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes).to(device)
    outputs = model.forward(tensor)
    return outputs[0].tolist()


def batch_prediction(image_bytes_batch):
    image_tensors = [transform_image(image_bytes=image_bytes) for image_bytes in image_bytes_batch]
    tensor = torch.cat(image_tensors).to(device)
    outputs = model.forward(tensor)
    return outputs.tolist()


if __name__ == "__main__":
    with open("113.82103_22.67296591_0_0.png", 'rb') as f:
        image_bytes = f.read()

    result = get_prediction(image_bytes)
    print(result)
    batch_result = batch_prediction([image_bytes] * 64)
    assert batch_result == [result] * 64
