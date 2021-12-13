import paddle
import paddle.nn as nn
paddle.set_device('cpu')


class Attention(nn.Layer):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, dropout=0., attention_dropout=0.):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        self.qkv = nn.Linear(embed_dim,
                             self.all_head_dim * 3,
                             bias_attr=False if qkv_bias is False else None
                             )
        self.scale = self.head_dim ** -0.5 if qk_scale is None else qk_scale
        self.softmax = nn.Softmax(-1)
        self.proj = nn.Linear(self.all_head_dim, embed_dim)

    def transpose_multi_head(self, x):
        # [B, N, all_head_dim]
        new_shape = x.shape[:-1] + [self.num_heads, self.head_dim]
        x = x.reshape(new_shape)
        # print('check', x.shape)
        #x [B, N, num_heads, head_dim]
        x = x.transpose([0, 2, 1, 3])
        # x [B, num_heads, num_patches, head_dim]
        return x

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).chunk(3, -1)
        # [B, N, all_head_dim] * 3
        q, k, v = map(self.transpose_multi_head, qkv)
        # q,k,v: [B, num_heads, num_patches, head_dim]
        attn = paddle.matmul(q, k, transpose_y=True)  # q * k^t
        attn = self.scale * attn
        attn = self.softmax(attn)
        # dropout
        # attn [B, num_heads, num_patches, num_patches]

        out = paddle.matmul(attn, v) # softmax(scale*(q*k')) * v
        # out [B, num_heads, num_patches, head_dim]
        out = out.transpose([0, 2, 1, 3])
        # out [B, num_patches, num_heads, head_dim]
        out = out.reshape([B, N, -1])
        out = self.proj(out)
        # dropout
        return out


def main():
    t = paddle.randn([8, 16, 96])
    model = Attention(embed_dim=96, num_heads=4, qkv_bias=False, qk_scale=None)
    print(model)
    out = model(t)
    print(out.shape)

if __name__ == '__main__':
    main()