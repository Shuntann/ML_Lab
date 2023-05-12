import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


#画像表示用、縦横5x5で25枚の表示
def show_tensor_images(i,image_flattened, num_images=25, size=(1, 28, 28)):
  image = image_flattened.detach().cpu().view(-1, *size) # 画像のサイズ1x28x28に戻す
  image_grid = make_grid(image[:num_images], nrow=5) # 画像を並べる
  plt.imshow(image_grid.permute(1, 2, 0).squeeze()) # 画像の表示
  plt.savefig(f"results/result_epoch{i}.png")
  print(f"finished epoch{i}")
  plt.show()



# ganerator convTrans → batchNorm → ReLU
# last layer → convTrans→ Tanh

'''バッチ正規化を使用。
プーリング処理は使わず、ストライドを使った畳み込みを利用する。
活性化関数にReLUを使う。最後の層はTanhを使う。'''

class GeneratorBlock(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size=3, stride=2,
                    final_layer=False):
    super(GeneratorBlock, self).__init__()
    if not final_layer:
      self.generator_block = nn.Sequential(
          nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride), #convTransopose
          nn.BatchNorm2d(output_channels),#バッチ正規化
          nn.ReLU(inplace=True))#活性化関数ReLU
    else:
      self.generator_block = nn.Sequential(# last layer
          nn.ConvTranspose2d(input_channels, output_channels,
                             kernel_size, stride),# convTranspose
                             nn.Tanh()) #活性化関数Tanh
  def forward(self, x):
    return self.generator_block(x)


# 最初にチャンネル数を確定し、そこから半分ずつにアウトプットを設定することでlast layerではチャンネル数1となる

class Generator(nn.Module):
  def __init__(self, z_dim=10, image_dim=1, hidden_dim=128):
    super(Generator, self).__init__()
    self.z_dim = z_dim
    self.generator = nn.Sequential(GeneratorBlock(z_dim, hidden_dim * 4), #インプット：ノイズベクトル10次元、隠れ層128, チャンネル数４
                                   GeneratorBlock(hidden_dim * 4, hidden_dim * 2,# インプット:128x4, 隠れ層:128x2,
                                                  kernel_size=4, stride=1),
                                   GeneratorBlock(hidden_dim * 2, hidden_dim),#
                                   GeneratorBlock(hidden_dim, image_dim,
                                                  kernel_size=4, final_layer=True)) #lastlayer

  #生成過程
  def forward(self, noise):
    noise_reshaped = noise.view(len(noise), self.z_dim, 1, 1)
    return self.generator(noise_reshaped)

  def get_generator(self):
    return self.generator


#入力用のノイズを生成する関数
def get_noise(n_samples, z_dim, device='cuda'):
  return torch.randn(n_samples, z_dim, device=device)#ランダムノイズの生成関数
#n_samples = number of real_images, z_dim=64

# discriminator
# conv →batchNorm → LeakyRELU

class DiscriminatorBlock(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
    super(DiscriminatorBlock, self).__init__()
    if not final_layer:
      self.discriminator_block = nn.Sequential(nn.Conv2d(input_channels, output_channels,
                                                        kernel_size, stride),
                                              nn.BatchNorm2d(output_channels),
                                              nn.LeakyReLU(negative_slope=0.05, # alpha = 0.2→変更１0.01 →変更２0.05
                                                           inplace=True))
    else:
      self.discriminator_block = nn.Sequential(nn.Conv2d(input_channels, output_channels,
                                                         kernel_size, stride))
  def forward(self, x):
    return self.discriminator_block(x)

class Discriminator(nn.Module):
  def __init__(self, image_channels, hidden_channels):
    super(Discriminator, self).__init__()
    self.discriminator = nn.Sequential(DiscriminatorBlock(image_channels, hidden_channels),
                                       DiscriminatorBlock(hidden_channels, hidden_channels * 2),
                                       DiscriminatorBlock(hidden_channels * 2, 1,
                                                          final_layer=True))
  #推論過程
  def forward(self, input_images):
    prediction = self.discriminator(input_images)
    return prediction.view(len(prediction), -1)

# ウェイトの初期化
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

