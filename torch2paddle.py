import numpy as np
import torch
import paddle

from models.ChangeFormer import ChangeFormerV6

def torch2paddle():
    torch_path = "./pretrained_changeformer/pretrained_changeformer.pt"
    paddle_path = "./pretrained_changeformer/pretrained_changeformer.pdparam"
    torch_state_dict = torch.load(torch_path,map_location=torch.device('cpu'))
    torch_model_state_dict = torch_state_dict['model_G_state_dict']
    print(torch_state_dict.keys())

    model = ChangeFormerV6(embed_dim=256)
    print(model.state_dict().keys())
    model_state_dict = model.state_dict()
    # fc_names = ["classifier"]
    paddle_state_dict = torch_state_dict

    paddle_model_state_dict = {}

    layers_index = {}
    for name, param in model.named_sublayers():
        layers_index[name] = param

    for k in torch_model_state_dict:
        pd_k = k
        pd_k = pd_k.replace("running_var", "_variance")
        pd_k = pd_k.replace("running_mean", "_mean")

        if "num_batches_tracked" in k:
            continue

        v = torch_model_state_dict[k].detach().cpu().numpy()
        pd_v = model_state_dict[pd_k].detach().cpu().numpy()
        # print(pd_v)
        # print(pd_k)
        # print(pd_k.rsplit(".",1)[0])
        # print(layers_index[pd_k.rsplit(".",1)[0]])
        # layer = getattr(model, "Tenc_x2.patch_embed1.proj")
        # print(pd_k, isinstance(layer,paddle.nn.Linear))

        # print("v_shape:", v.shape)
        # print("pd_v_shape:", pd_v.shape)



        # if v.shape != pd_v.shape:
        if isinstance(layers_index[pd_k.rsplit(".",1)[0]], paddle.nn.Linear) and pd_k.rsplit(".",1)[1] == "weight":
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)

        paddle_model_state_dict[pd_k] = v

    paddle_state_dict['model_G_state_dict'] = paddle_model_state_dict
    del paddle_state_dict['optimizer_G_state_dict']
    del paddle_state_dict['exp_lr_scheduler_G_state_dict']
    paddle.save(paddle_state_dict, paddle_path)

if __name__ == "__main__":
    torch2paddle()