# ViT Online Class

import paddle
import numpy as np
from PIL import Image

paddle.set_device('cpu')

def main():
    # 1.create a tensor
    # t = paddle.zeros([3, 3])
    # print(t)

    # 2. randn tensor
    # t = paddle.randn([5, 3])
    # print(t)

    # 3. image
    # img = np.array(Image.open('./724.jpg'))
    img = paddle.randint(0, 255, [28, 28])
    img = np.array(img)
    # for i in range(28):
    #     for j in range(28):
    #         print(f'{img[i, j]:03} ', end='')
    #     print()
    t = paddle.to_tensor(img, dtype='float32')

    # 4.tensor type, dtype
    # print(type(t))
    # print(t.dtype)

    # 5. transpose image tensor
    # print(t.transpose([1, 0]))

    # 6.reshape

    # 7.unsqueeze

    # 8.chunk
    t = paddle.randint(0, 10, [5, 15])
    qkv = t.chunk(3, -1)
    print(type(qkv))
    q, k, v = qkv
    print(t)
    print(q)

if __name__ == '__main__':
    main()