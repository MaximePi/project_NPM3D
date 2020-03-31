#python3 main.py --exp_name grid_sub_max_512 --epochs 50 --num_points 512 --aggregation 'max' --grid_samp True
python3 main.py --exp_name grid_sub_max_512 --aggregation 'max' --eval True --grid_samp True --model_path 'checkpoints/grid_sub_max_512/models/model.t7'
#python3 main.py --exp_name grid_sub_max_1024 --epochs 50 --num_points 1024 --aggregation 'max' --grid_samp True
python3 main.py --exp_name grid_sub_max_1024 --aggregation 'max' --eval True --grid_samp True --model_path 'checkpoints/grid_sub_max_1024/models/model.t7'
#python3 main.py --exp_name grid_sub_max_1536 --epochs 50 --num_points 1536 --aggregation 'max' --grid_samp True
python3 main.py --exp_name grid_sub_max_1536 --aggregation 'max' --eval True --grid_samp True --model_path 'checkpoints/grid_sub_max_1536/models/model.t7'
echo "mean"
#python3 main.py --exp_name grid_sub_mean_512 --epochs 50 --num_points 512 --aggregation 'mean' --grid_samp True
python3 main.py --exp_name grid_sub_mean_512 --eval True --aggregation 'mean' --grid_samp True --model_path 'checkpoints/grid_sub_mean_512/models/model.t7'
#python3 main.py --exp_name grid_sub_mean_1024 --epochs 50 --num_points 1024 --aggregation 'mean' --grid_samp True
python3 main.py --exp_name grid_sub_mean_1024 --eval True  --aggregation 'mean' --grid_samp True --model_path 'checkpoints/grid_sub_mean_1024/models/model.t7'
#python3 main.py --exp_name grid_sub_mean_1536 --epochs 50 --num_points 1536 --aggregation 'mean' --grid_samp True
python3 main.py --exp_name grid_sub_mean_1536 --eval True --aggregation 'mean'  --grid_samp True --model_path 'checkpoints/grid_sub_mean_1536/models/model.t7'
echo "sum"
#python3 main.py --exp_name grid_sub_sum_512 --epochs 50 --num_points 512 --aggregation 'sum' --grid_samp True
#python3 main.py --exp_name grid_sub_sum_512 --eval True --model_path 'checkpoints/grid_sub_sum_512/models/model.t7'
#python3 main.py --exp_name grid_sub_sum_1024 --epochs 50 --num_points 1024 --aggregation 'sum' --grid_samp True
#python3 main.py --exp_name grid_sub_sum_1024 --eval True --model_path 'checkpoints/grid_sub_sum_1024/models/model.t7'
#python3 main.py --exp_name grid_sub_sum_1536 --epochs 50 --num_points 1536 --aggregation 'sum' --grid_samp True
#python3 main.py --exp_name grid_sub_sum_1536 --eval True --model_path 'checkpoints/grid_sub_sum_1536/models/model.t7'
