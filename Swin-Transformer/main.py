import paddle
import paddle.nn as nn
# import os
# import sys
# sys.path.join('../')
from mask import generate_mask

paddle.set_device('cpu')


class PatchEmbedding(nn.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super(PatchEmbedding, self).__init__()
        self.patch_embed = nn.Conv2D(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # [n, embed_dim, h', w']
        x = x.flatten(2)  # [n, embed_dim, h'*w']
        x = x.transpose([0, 2, 1])  # [n, h'*w', embed_dim]
        x = self.norm(x)
        return x


class PatchMerging(nn.Layer):
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.resolution
        b, _, c = x.shape

        x = x.reshape([b, h, w, c])

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = paddle.concat([x0, x1, x2, x3], axis=-1)  # [b, h/2, w/2, 4c]
        x = x.reshape([b, -1, 4 * c])
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Mlp(nn.Layer):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    # [B, h//ws, w//ws, ws, ws, c]
    x = x.reshape([-1, window_size, window_size, C])
    # [B * num_patches, ws, ws, C]
    return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] // (H / window_size * W / window_size))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5])
    x = x.reshape([B, H, W, -1])
    return x

class WindowAttention(nn.Layer):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(-1)
        self.qkv = nn.Linear(dim,
                             dim * 3)
        self.proj = nn.Linear(dim, dim)

        # relative position bias
        self.window_size = window_size
        self.relative_position_bias_table = paddle.create_parameter(
            shape=[(2*window_size-1)*(2*window_size-1), num_heads],
            dtype='float32',
            default_initializer=nn.initializer.TruncatedNormal(std=.02))
        # print('check:', self.relative_position_bias_table)
        coord_h = paddle.arange(self.window_size)
        coord_w = paddle.arange(self.window_size)
        coords = paddle.stack(paddle.meshgrid([coord_h, coord_w]))  # [2, ws, ws]
        coords = coords.flatten(1)   # [2, ws*ws]
        relative_coords = coords.unsqueeze(2) - coords.unsqueeze(1)
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1

        relative_coords[:, :, 0] *= 2*self.window_size - 1
        relative_coords_index = relative_coords.sum(2)
        print('relative_coords_index', relative_coords_index)
        self.register_buffer('relative_coords_index', relative_coords_index)

    def get_relative_position_bias_from_index(self):
        table = self.relative_position_bias_table  # [2m-1 * 2m-1, num_heads]
        index = self.relative_coords_index.reshape([-1]) # [M^2, M^2] - > [M^2*M^2]
        relative_position_bias = paddle.index_select(x=table, index=index) # [M*M, M*M, num_heads]
        return relative_position_bias

    def transpose_multi_head(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)      # [B, num_patches, num_heads, dim_head]
        x = x.transpose([0, 2, 1, 3])   # [B, num_heads, num_patches, dim_head]
        return x

    def forward(self, x, mask=None):
        # x: [B, num_patches, embed_dim]
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        q, k, v = map(self.transpose_multi_head, qkv)

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        # attn = self.softmax(attn)

        ###### BEGIN Class 6: Relative Position Bias
        relative_position_bias = self.get_relative_position_bias_from_index()
        relative_position_bias = relative_position_bias.reshape(
            [self.window_size * self.window_size, self.window_size * self.window_size, -1])
        # [num_patches, num_patches, num_heads]
        relative_position_bias = relative_position_bias.transpose([2, 0, 1]) #[num_heads, num_patches, num_patches]
        # attn: [B*num_windows, num_heads, num_patches, num_patches]
        attn = attn + relative_position_bias.unsqueeze(0)
        ###### END Class 6: Relative Position Bias

        # mask
        if mask is None:
            attn = self.softmax(attn)
        else:
            # mask: [num_windows, num_patches, num_patches]
            # attn: [B*num_windows, num_heads, num_patches, num_patches]
            attn = attn.reshape([B // mask.shape[0], mask.shape[0], self.num_heads, mask.shape[1], mask.shape[1]])
            # attn: [B, num_windows, num_heads, num_patches, num_patches]
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # mask: [1, num_windows, 1, num_patches, num_patches]
            attn = attn.reshape([-1, self.num_heads, mask.shape[1], mask.shape[1]])
            # attn: [B*num_windows, num_heads, num_patches, num_patches]

        out = paddle.matmul(attn, v)   # [B, num_heads, num_patches, dim_head]
        out = out.transpose([0, 2, 1, 3])
        # [B, num_patches, num_heads, dim_head]  num_head * dim_head = embed_dim
        out = out.reshape([B, N, C])
        out = self.proj(out)
        return out

class SwinBlock(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.resolution = input_resolution
        self.window_size = window_size

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

        if self.shift_size > 0:
            attn_mask = generate_mask(self.window_size, self.shift_size, self.resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)  # 将其注册到网络中

    def forward(self, x):
        H, W = self.resolution
        B, N, C = x.shape

        h = x
        x = self.attn_norm(x)

        x = x.reshape([B, H, W, C])

        ### CLASS 6
        # shift window
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x
        # compute window attn
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # shift back
        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        x_windows = window_partition(x, self.window_size)
        # [B * num_patches, ws, ws, C]
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        x = window_reverse(attn_windows, self.window_size, H, W)
        # [B, H, W, C]
        x = x.reshape([B, H*W, C])
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x

def main():
    # class 5
    # t = paddle.randn([4, 3, 224, 224])
    # patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    # swin_block = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7)
    # patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)
    # print('image shape out shape : [4, 3, 224, 224]')
    # out = patch_embedding(t)  # [4, 56*56, 96]
    # print('patch_embedding out shape:', out.shape)
    # out = swin_block(out)  # [4, 56*56, 96]
    # print('swin_embedding out shape:', out.shape)
    # out = patch_merging(out)   # [4, 28*28, 192]
    # print('patch_merging out shape:', out.shape)

    t = paddle.randn((4, 3, 224, 224))
    patch_embedding = PatchEmbedding(patch_size=4, embed_dim=96)
    swin_block_w_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=0)
    swin_block_sw_msa = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7, shift_size=7//2)
    patch_merging = PatchMerging(input_resolution=[56, 56], dim=96)

    print('image shape = [4, 3, 224, 224]')
    out = patch_embedding(t)  # [4, 56, 56, 96]
    print('patch_embedding out shape = ', out.shape)
    out = swin_block_w_msa(out)
    out = swin_block_sw_msa(out)
    print('swin_block out shape = ', out.shape)
    out = patch_merging(out)
    print('patch_merging out shape = ', out.shape)

if __name__ == '__main__':
    main()
