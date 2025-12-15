
# base
python /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/eval_lao_models.py \
    --data /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/pond4_lao_offline_data.npz \
    --ckpt /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/saved_models/LAO_nets/lao_models.pt \
    --hidden 128

# base- large
python /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/eval_lao_models.py \
    --data /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/pond4_lao_offline_data.npz \
    --ckpt /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/saved_models/LAO_nets/lao_models_large.pt \
    --hidden 256


# #real case
# python /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/eval_lao_models.py \
#     --data /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/pond4_lao_offline_data_real_case.npz \
#     --ckpt /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/saved_models/LAO_nets/lao_models_real_case.pt \
#     --hidden 256 \