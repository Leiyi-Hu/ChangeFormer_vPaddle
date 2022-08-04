# split images for predict
import sys
sys.path.append('.')
import os
import glob
import argparse

from paddle import inference
import numpy as np
import paddle as pd
import paddle.vision.transforms as TF
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        '-m',
        type=str,
        default=None,
        help='model directory path')
    parser.add_argument(
        '--img_dir',
        '-s',
        type=str,
        default=None,
        help='path to save inference model')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./test_tipc/output/infer_result/',
        help='path to save inference result')

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=None)

    return parser


class InferenceEngine(object):
    """InferenceEngine
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args
        # init inference engine
        self.predictor, self.config = self.load_predictor(
            os.path.join(args.model_dir, "model.pdmodel"),
            os.path.join(args.model_dir, "model.pdiparams"))    

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        config = inference.Config(model_file_path, params_file_path)
        if self.args.use_gpu:
            config.enable_use_gpu(args.gpu_mem, 0) # 使用GPU
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(1)
        
        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config

    def preprocess(self, imgfiles):
        """preprocess
        Preprocess to the input.
        Args:
            imgfiles: A B image path to change detect.
        Returns: Input data after preprocess.
        """
        A_path, B_path = imgfiles

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        imgs = [img, img_B]

        imgs = [pd.Tensor(np.array(img, np.float32)).transpose(
                [2, 0, 1]) / 255.0 for img in imgs]   
        imgs = [
                TF.normalize(
                    img, mean=[
                        0.5, 0.5, 0.5], std=[
                        0.5, 0.5, 0.5], data_format="CHW") for img in imgs]

        [img, img_B] = imgs
        preprocessed_samples = {
                'image': np.expand_dims(img.numpy(), 0),
                'image2': np.expand_dims(img_B.numpy(),0)
            }
        return preprocessed_samples

    def raw_predict(self, inputs):
        """ 接受预处理过后的数据进行预测
            Args:
                inputs(dict): 预处理过后的数据
        """
        input_names = self.predictor.get_input_names()
        for name in input_names:
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(inputs[name])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        net_outputs = list()
        for name in output_names:
            output_tensor = self.predictor.get_output_handle(name)
            net_outputs.append(output_tensor.copy_to_cpu())

        return net_outputs[-1]

    def postprocess(self, G_pred):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            G_pred: last one output from the ChangeFormer
        Returns: Output denoised image.
        """
        pred = np.argmax(G_pred, axis=1)
        pred_vis = pred * 255     
        return pred_vis[0]


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    inference_engine = InferenceEngine(args)

    img_dir = args.img_dir
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    prefix = '.png'
    floder1 = os.path.join(img_dir, 'A')
    floder2 = os.path.join(img_dir, 'B')

    img_list1 = [f for f in os.listdir(floder1) if f.endswith(prefix)]
    print('total file number is {}'.format(len(img_list1)))
    for filename in img_list1:
        imgfile = (os.path.join(floder1, filename), os.path.join(floder2,filename))
        inputs = inference_engine.preprocess(imgfile)
        net_outputs = inference_engine.raw_predict(inputs)
        result = inference_engine.postprocess(net_outputs)
        image_pil = Image.fromarray(np.array(result, dtype=np.uint8))
        image_pil.save(os.path.join(output_dir, filename))

    print('finish')