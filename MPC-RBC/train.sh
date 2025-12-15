
# base
python /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/train_lao_nets.py \
    --data /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/pond4_lao_offline_data.npz \
    --out /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/saved_models/LAO_nets/lao_models_large.pt \
    --hidden 256 \
    --epochs 5000 \
    --large_model



# #real case
# python /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/train_lao_nets.py \
#     --data /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/pond4_lao_offline_data_real_case.npz \
#     --out /home/mila/y/yuns/project/HW/IFT6162_Project/MPC-RBC/saved_models/LAO_nets/lao_models_real_case.pt \
#     --hidden 256 \
#     --epochs 10000