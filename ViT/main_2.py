import numpy as np
import paddle
import paddle.nn as nn
from PIL import Image
paddle.set_device('cpu')


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class PatchEmbedding(nn.Layer):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super(PatchEmbedding, self).__init__()
        self.patch_embed = nn.Conv2D(in_channels,
                                     embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size,
                                     weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.0)),
                                     bias_attr=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x [1, 1, 28, 28]
        x = self.patch_embed(x)
        # x [n, c', h', w']
        x = x.flatten(2)   #[n, embed_dim, h'*w']
        x = x.transpose([0, 2, 1])  #[n, h'*w', embed_dim]
        x = self.dropout(x)
        return x



def main():
    # 1.load image
    img = np.array(paddle.randint(0, 255, [28, 28]))
    sample = paddle.to_tensor(img, dtype='float32')
    # simulate a batch of data
    sample = sample.reshape([1, 1, 28, 28])
    print(sample.shape)

    # 2. Patch embedding
    patch_embed = PatchEmbedding(image_size=28, patch_size=7, in_channels=1, embed_dim=1)
    out = patch_embed(sample)
    print(out.shape)


    # 3. MLP
    mlp = Mlp(1)
    out = mlp(out)
    print('out.shape=', out.shape)


if __name__ == '__main__':
    main()