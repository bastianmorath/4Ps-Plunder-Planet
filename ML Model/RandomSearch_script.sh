
# This script tries out different window sizes and sends a job to the Euler-cluster which
# does RandomSearchCV for all classifiers

# for gradient_w in 2 5 10 30
# do
#     for hw in 2  10 30 60
#     do
#         for cw in 2  10 30 60
#         do
#             for (( clf_idx=0; clf_idx <=6; clf_idx ++))
#             do
#                 bsub -W 100:00 -N 'python "$PWD"/hyperparameter_optimization.py $clf_idx 200 $hw $cw gradient_w'
#             done
#         done
#     done
# done

for clf_idx in `seq 0 6`
do
    bsub -W 240:00 python "$PWD"/hyperparameter_optimization.py $clf_idx 200
done
