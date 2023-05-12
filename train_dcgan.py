import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from parameters import *


z_dim = 64
batch_size = 128
learning_rate = 0.005#→0.0002→変更１0.002→変更２0.005
beta_1 = 0.5
beta_2 = 0.999
num_of_epochs = 10
device = 'cuda'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, ), (0.5, ))])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)


image_channels = 1
hidden_channles = 16
# インスタンス化
generator = Generator(z_dim).to(device)
discriminator = Discriminator(image_channels=image_channels, hidden_channels=hidden_channles).to(device)

# オプティマイザ
gen_opt = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta_1, beta_2))



generator = generator.apply(weights_init)
discriminator = discriminator.apply(weights_init)

criterion = nn.BCEWithLogitsLoss() #criterion(estimate, real)で使う

for i,epoch in enumerate(range(num_of_epochs), start=1):
  mean_generator_loss = 0 #1エポックごとに損失関数の初期化
  mean_discriminator_loss = 0
  for real_images, _ in tqdm(dataloader):#バッチサイズ分画像を抽出（このfor文では64回回る＝1epochでは64枚の画像を精査)
    real_images = real_images.to(device) #imagesをGPUへ

    # discriminator
    disc_opt.zero_grad() # 勾配の初期化
    # 偽画像
    noise = get_noise(len(real_images), z_dim, device=device) # ノイズの生成
    fake_images = generator(noise) # 偽画像を生成
    disc_fake_prediction = discriminator(fake_images.detach()) # Discriminatorでfake画像を予測
    correct_labels = torch.zeros_like(disc_fake_prediction) # 偽画像の正解ラベルは0
    disc_fake_loss = criterion(disc_fake_prediction, correct_labels) # 偽画像に対する損失を計算 log(1-(dis_fake_prediction))

    # 本物の画像
    disc_real_prediction = discriminator(real_images) # Discriminatorで予測
    correct_labels = torch.ones_like(disc_real_prediction) # 本物の画像の正解ラベルは1
    disc_real_loss = criterion(disc_real_prediction, correct_labels) # 本物の画像に対する損失を計算　最適値は0.6931

    # 最終的な損失
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    disc_loss.backward()#勾配計算
    disc_opt.step()#学習

    # エポックごとの損失
    mean_discriminator_loss += disc_loss / len(real_images)#64個分 最適値は0.6931

    # generator
    gen_opt.zero_grad() # 勾配の初期化
    fake_noise = get_noise(len(real_images), z_dim, device=device) # ノイズの生成
    fake_images = generator(fake_noise) # 偽画像の生成
    disc_fake_prediction = discriminator(fake_images) # Discriminatorで予測
    correct_labels = torch.ones_like(disc_fake_prediction) # 本物の正解ラベルは1
    gen_loss = criterion(disc_fake_prediction, correct_labels) # 損失を計算　(1-(t=0))log(1-y)
    gen_loss.backward()#勾配の計算
    gen_opt.step()#学習
    # エポックごとの損失
    mean_generator_loss += gen_loss / len(real_images)

    # #accuracy計算
    # disc_fake_acc = torch.sum(disc_fake_prediction) / batch_size
    # disc_real_acc = torch.sum(disc_real_prediction) / batch_size


    # mean_gen_acc = (batch_size - torch.sum(disc_fake_prediction)) / batch_size


  print(f'Generator loss: {mean_generator_loss}') #最適値は0.6931
  print(f'Discriminator loss: {mean_discriminator_loss}')# 最適値は0.6931　(または0.3466????)

  # print(f'Generator accuracy: {mean_gen_acc*100} %')
  # print(f'Discriminator fake accuracy: {disc_fake_acc*100} %')
  # print(f'Discriminator real accuracy: {disc_real_acc*100} %')



  # 生成される画像を表示
  # noise = get_noise(len(real_images), z_dim, device=device)
  show_tensor_images(i,generator(noise))
  print(noise[:25])
  

torch.save(discriminator, "discriminator.pth")
torch.save(generator, "generator.pth")