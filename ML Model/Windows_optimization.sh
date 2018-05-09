
# This script tries out different window sizes and sends a job to the Euler-cluster which
# does RandomSearchCV for all classifiers


for gradient_w in 2 5 10
do
    for hw in 2 10 30 60
    do
        for cw in 2 10 30 60
        do
            bsub -W 240:00 python "$PWD"/window_optimization.py $hw $cw $gradient_w
        done
    done
done
