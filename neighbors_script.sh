echo "128"
#python3 main.py --exp_name num_128_ --epochs 50 --num_points 128
#python3 main.py --exp_name num_128 --eval True --num_points 128 --model_path 'checkpoints/num_128/models/model.t7'
echo "256"
#python3 main.py --exp_name num_256 --epochs 50 --num_points 256
#python3 main.py --exp_name num_256 --eval True --num_points 256 --model_path 'checkpoints/num_256/models/model.t7'
echo "512"
#python3 main.py --exp_name num_512 --epochs 50 --num_poins 512
#python3 main.py --exp_name num_512 --eval True --num_points 512 --model_path 'checkpoints/num_512/models/model.t7'
echo "1024"
#python3 main.py --exp_name num_1024 --epochs 50 --num_points 1024
#python3 main.py --exp_name num_1024 --eval True --num_points 1024 --model_path 'checkpoints/num_1024/models/model.t7'
echo "1536"
#python3 main.py --exp_name num_1536 --epochs 50 --num_points 1536
#python3 main.py --exp_name num_1536 --eval True --num_points 1536 --model_path 'checkpoints/num_1536/models/model.t7'
echo "2024"
#python3 main.py --exp_name num_2048 --epochs 50 --num_points 2048
#python3 main.py --exp_name num_2048 --eval True --num_points 2048 --model_path 'checkpoints/num_2048/models/model.t7'
echo "256"
#python3 main.py --exp_name emb_256 --epochs 50 --emb_dims 256
#python3 main.py --exp_name emb_256 --eval True --emb_dims 256 --model_path 'checkpoints/emb_256/models/model.t7'
echo "512"
#python3 main.py --exp_name emb_512 --epochs 50 --emb_dims 512
#python3 main.py --exp_name emb_512 --eval True --emb_dims 512 --model_path 'checkpoints/emb_512/models/model.t7'
echo "1024"
#python3 main.py --exp_name emb_1024 --epochs 50 --emb_dims 1024
#python3 main.py --exp_name emb_1024 --eval True --emb_dims 1024 --model_path 'checkpoints/emb_1024/models/model.t7'
echo "1536"
#python3 main.py --exp_name emb_1536 --epochs 50 --emb_dims 1536
#python3 main.py --exp_name emb_1536 --eval True --emb_dims 1536 --model_path 'checkpoints/emb_1536/models/model.t7'
echo "2024"
#python3 main.py --exp_name emb_2048 --epochs 50 --emb_dims 2048
#python3 main.py --exp_name emb_2048 --eval True --emb_dims 2048 --model_path 'checkpoints/emb_2048/models/model.t7'

#python3 main.py --exp_name K5 --eval True --k 5 --model_path 'checkpoints/K5/models/model.t7'
#python3 main.py --exp_name K10 --eval True --k 10 --model_path 'checkpoints/K10/models/model.t7'
#python3 main.py --exp_name K15 --eval True --k 15 --model_path 'checkpoints/K15/models/model.t7'
#python3 main.py --exp_name K20 --eval True --k 20 --model_path 'checkpoints/K20/models/model.t7'
#python3 main.py --exp_name K30 --eval True --k 30 --model_path 'checkpoints/K30/models/model.t7'
#python3 main.py --exp_name K40 --eval True --k 40 --model_path 'checkpoints/K40/models/model.t7'

#python3 main.py --exp_name K5_mean --aggregation 'mean' --eval True --k 5 --model_path 'checkpoints/K5_mean/models/model.t7'
#python3 main.py --exp_name K10_mean --aggregation 'mean' --eval True --k 10 --model_path 'checkpoints/K10_mean/models/model.t7'
#python3 main.py --exp_name K15_mean --aggregation 'mean' --eval True --k 15 --model_path 'checkpoints/K15_mean/models/model.t7'
#python3 main.py --exp_name K20_mean --aggregation 'mean' --eval True --k 20 --model_path 'checkpoints/K20_mean/models/model.t7'
#python3 main.py --exp_name K30_mean --aggregation 'mean' --eval True --k 30 --model_path 'checkpoints/K30_mean/models/model.t7'
python3 main.py --exp_name K40_mean --aggregation 'mean' --test_batch_size 8 --eval True --k 40 --model_path 'checkpoints/K40_mean/models/model.t7'

#python3 main.py --exp_name K5_sum --aggregation 'sum' --eval True --k 5 --model_path 'checkpoints/K5_sum/models/model.t7'
#python3 main.py --exp_name K10_sum --aggregation 'sum' --eval True --k 10 --model_path 'checkpoints/K10_sum/models/model.t7'
#python3 main.py --exp_name K15_sum --aggregation 'sum' --eval True --k 15 --model_path 'checkpoints/K15_sum/models/model.t7'
#python3 main.py --exp_name K20_sum --aggregation 'sum' --eval True --k 20 --model_path 'checkpoints/K20_sum/models/model.t7'
#python3 main.py --exp_name K30_sum --aggregation 'sum' --eval True --k 30 --model_path 'checkpoints/K30_sum/models/model.t7'
python3 main.py --exp_name K40_sum --aggregation 'sum' --test_batch_size 8 --eval True --k 40 --model_path 'checkpoints/K40_sum/models/model.t7'



