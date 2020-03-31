echo "Train"
#python3 main.py --exp_name 1024maxF --aggregation 'max' --add_features True --epochs 50 --batch_size 32 --test_batch_size 16 --num_points 1024
#python3 main.py --exp_name 512maxF --aggregation 'max' --add_features True --epochs 50 --batch_size 32 --test_batch_size 16 --num_points 512
#python3 main.py --exp_name 1536maxF --aggregation 'max' --add_features True --epochs 50 --batch_size 32 --test_batch_size 16 --num_points 1536
#python3 main.py --exp_name 1024meanF --aggregation 'mean' --add_features True --epochs 50 --batch_size 32 --test_batch_size 16 --num_points 1024
#python3 main.py --exp_name 512meanF --aggregation 'mean' --add_features True --epochs 50 --batch_size 32 --test_batch_size 16 --num_points 512
#python3 main.py --exp_name 1536meanF --aggregation 'mean' --add_features True --epochs 50 --batch_size 32 --test_batch_size 16 --num_points 1536
python3 main.py --exp_name 1024pnetF --model 'pointnet' --add_features True --epochs 50 --batch_size 16 --test_batch_size 16 --num_points 1024
python3 main.py --exp_name 1024pplusF --model 'pointnetplus' --add_features True --epochs 50 --batch_size 16 --test_batch_size 16 --num_points 1024
python3 main.py --exp_name 1024monetF --model 'monet' --add_features True --epochs 50 --batch_size 16 --test_batch_size 16 --num_points 1024

echo "Test"
#python3 main.py --exp_name 1024maxF --aggregation 'max' --add_features True --num_points 1024 --eval True --model_path 'checkpoints/1024maxF/models/model.t7'
#python3 main.py --exp_name 512maxF --aggregation 'max' --add_features True --num_points 512 --eval True --model_path 'checkpoints/536maxF/models/model.t7'
#python3 main.py --exp_name 1536maxF --aggregation 'max' --add_features True --num_points 1536 --eval True --model_path 'checkpoints/1536maxF/models/model.t7'
#python3 main.py --exp_name 1024meanF --aggregation 'mean' --add_features True --num_points 1024 --eval True --model_path 'checkpoints/1024meanF/models/model.t7'
#python3 main.py --exp_name 512meanF --aggregation 'mean' --add_features True --num_points 512 --eval True --model_path 'checkpoints/512meanF/models/model.t7'
#python3 main.py --exp_name 1536meanF --aggregation 'mean' --add_features True --num_points 1536 --eval True --model_path 'checkpoints/1536meanF/models/model.t7'
