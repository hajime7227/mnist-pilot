import os

import torch
import torchvision.transforms as transforms

from .configs.train_settings import get_args
from .models.cnn import Net

MODEL_PATH = "backend/saved_model/mnist_model.pth"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"モデルファイル {MODEL_PATH} が見つかりません")

    try:
        args = get_args(args=[])  # `argparse` の影響を避けるために空のリストを渡す
    except SystemExit:
        args = None  # 例外が発生した場合、デフォルト値を使う

    if args is None:

        class Args:
            no_cuda = False
            no_mps = False

        args = Args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)

    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


def predict_digit(image_array):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    image_tensor = transform(image_array).unsqueeze(0)  # (1, 1, 28, 28) に変換

    with torch.no_grad():
        model = load_model()
        device = next(model.parameters()).device  # モデルのデバイスを取得
        image_tensor = image_tensor.to(device)  # 入力テンソルをモデルのデバイスに移動
        output = model(image_tensor)
        predicted_digit = torch.argmax(output, dim=1).item()

    return predicted_digit
