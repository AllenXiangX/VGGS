import os

scenes = [83, 63, 37, 110, 24, 40, 106, 55, 65, 97, 114, 118, 122, 105, 69]  # 63 ] # [63, # []  # 110,  63] # , 

data_base_path='data/DTU/set_22_25_28'
config_path='configs/dtu.yaml'
out_base_path='exp/dtu'
eval_path='data/dtu_eval'
out_name='test'
gpu_id=0


for scene in scenes:
    
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = f'--quiet -r2 --ncc_scale 0.5 --config {config_path}'
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene}/dense -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = f'--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --config {config_path}'
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}"
    print(cmd)
    os.system(cmd)