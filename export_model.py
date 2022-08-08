import os
import sys
sys.path.append('.')

from argparse import ArgumentParser
import paddle
from paddle.static import InputSpec
from models.ChangeFormer import ChangeFormerV6

def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model_path', default='./ChangeFormer/best_ckpt.pdparams', type=str)
    parser.add_argument(
        '--save_inference_dir', default='./inference/', help='path where to save', type=str)

    # model
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    net_G = ChangeFormerV6()

    checkpoint = paddle.load(args.model_path)
    net_G.set_state_dict(checkpoint['model_G_state_dict'])

    net_G.eval()

    # 变化检测的输入是两张图

    image_shape = [1, 3, 256, 256]

    test_inputs = [InputSpec(shape=image_shape, name='image', dtype='float32'), 
    InputSpec(shape=image_shape, name='image2', dtype='float32')]

    static_net = paddle.jit.to_static(net_G, input_spec=test_inputs)
    paddle.jit.save(static_net, os.path.join(args.save_inference_dir, "model"))



