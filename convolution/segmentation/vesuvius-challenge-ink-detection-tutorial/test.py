# import sys
# sys.path.append('segment_anything')
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
# from functools import partial
# import torch
# from torchinfo import summary
# import tqdm

# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#     polygons = []
#     color = []
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         img = np.ones((m.shape[0], m.shape[1], 3))
#         color_mask = np.random.random((1, 3)).tolist()[0]
#         for i in range(3):
#             img[:,:,i] = color_mask[i]
#         ax.imshow(np.dstack((img, m*0.35)))
# in_chans = 6
# prompt_embed_dim = 256
# image_size = 224
# vit_patch_size = 16
# encoder_embed_dim=1024
# encoder_depth=24
# encoder_num_heads=16
# encoder_global_attn_indexes=[5, 11, 17, 23]
# image_embedding_size = image_size // vit_patch_size
# model = Sam(
#         image_encoder=ImageEncoderViT(
#             in_chans=in_chans,
#             depth=encoder_depth,
#             embed_dim=encoder_embed_dim,
#             img_size=image_size,
#             mlp_ratio=4,
#             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#             num_heads=encoder_num_heads,
#             patch_size=vit_patch_size,
#             qkv_bias=True,
#             use_rel_pos=True,
#             global_attn_indexes=encoder_global_attn_indexes,
#             window_size=14,
#             out_chans=prompt_embed_dim,
#         ),
#         mask_decoder=MaskDecoder(
#             num_multimask_outputs=1,
#             transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=prompt_embed_dim,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=prompt_embed_dim,
#             iou_head_depth=3,
#             iou_head_hidden_dim=256,
#         ),
#         device=torch.device('cuda')
#     ).cuda()

# print(model)
# summary(model, input_size=(6, 224, 224))
   
# # x = torch.zeros((8,6,224,224)).cuda()
# # o = model(x)
