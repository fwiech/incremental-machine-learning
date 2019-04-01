#!/bin/bash


echo "Task T2 on BS1000"
python3 src/main.py \
--classes 5 6 7 8 9 \
--learnrate 0.00001 \
--iterations 2500 \
--batch 100 \
--previous '55_A_BS1000' \
--batch_fisher 1000 \
-d 1000


# echo "P10-10"

# echo "train first network with BS1000"
# python3 src/main.py \
# --classes 0 1 2 3 4 5 6 7 8 9 \
# --learnrate 0.001 \
# --iterations 2500 \
# --batch 100 \
# --batch_fisher 1000 \
# -s 'PM_A_BS1000' \
# -d 50

# echo "recalculate the trained network with FIM"
# python3 src/main.py \
# --classes 0 1 2 3 4 5 6 7 8 9 \
# --learnrate 0.001 \
# --iterations 2500 \
# --batch 100 \
# --batch_fisher 1 \
# --previous 'PM_A_BS1000' \
# --notrain \
# -s 'PM_A_FIM' \
# -d 50

# echo "Task T2 on BS1000"
# python3 src/main.py \
# --classes 0 1 2 3 4 5 6 7 8 9 \
# --learnrate 0.00001 \
# --iterations 20000 \
# --batch 100 \
# --previous 'PM_A_BS1000' \
# --batch_fisher 1000 \
# --permute 0 \
# -s 'PM_B_BS1000' \
# -d 1000

# echo "Task T2 on FIM"
# python3 src/main.py \
# --classes 0 1 2 3 4 5 6 7 8 9 \
# --learnrate 0.00001 \
# --iterations 20000 \
# --batch 100 \
# --previous 'PM_A_FIM' \
# --batch_fisher 1000 \
# --permute 0 \
# -s 'PM_B_FIM' \
# -d 1000




# echo "D5-5"

# echo "train first network with BS1000"
# python3 src/main.py \
# --classes 0 1 2 3 4 \
# --learnrate 0.001 \
# --iterations 2500 \
# --batch 100 \
# --batch_fisher 1000 \
# -s '55_A_BS1000' \
# -d 50

# echo "recalculate the trained network with FIM"
# python3 src/main.py \
# --classes 0 1 2 3 4 \
# --learnrate 0.001 \
# --iterations 2500 \
# --batch 100 \
# --batch_fisher 1 \
# --previous '55_A_BS1000' \
# --notrain \
# -s '55_A_FIM' \
# -d 50

# echo "Task T2 on BS1000"
# python3 src/main.py \
# --classes 5 6 7 8 9 \
# --learnrate 0.00001 \
# --iterations 2500 \
# --batch 100 \
# --previous '55_A_BS1000' \
# --batch_fisher 1000 \
# -s '55_B_BS1000' \
# -d 50

# echo "Task T2 on FIM"
# python3 src/main.py \
# --classes 5 6 7 8 9 \
# --learnrate 0.00001 \
# --iterations 2500 \
# --batch 100 \
# --previous '55_A_FIM' \
# --batch_fisher 1000 \
# -s '55_B_FIM' \
# -d 50





# echo "D9-1"

# echo "train first network with BS1000"
# python3 src/main.py \
# --classes 0 1 2 3 4 5 6 7 8 \
# --learnrate 0.001 \
# --iterations 2500 \
# --batch 100 \
# --batch_fisher 1000 \
# -s '91_A_BS1000' \
# -d 50

# echo "recalculate the trained network with FIM"
# python3 src/main.py \
# --classes 0 1 2 3 4 5 6 7 8 \
# --learnrate 0.001 \
# --iterations 2500 \
# --batch 100 \
# --batch_fisher 1 \
# --previous '91_A_BS1000' \
# --notrain \
# -s '91_A_FIM' \
# -d 50

# echo "Task T2 on BS1000"
# python3 src/main.py \
# --classes 9 \
# --learnrate 0.00001 \
# --iterations 2500 \
# --batch 100 \
# --previous '91_A_BS1000' \
# --batch_fisher 1000 \
# -s '91_B_BS1000' \
# -d 50

# echo "Task T2 on FIM"
# python3 src/main.py \
# --classes 9 \
# --learnrate 0.00001 \
# --iterations 2500 \
# --batch 100 \
# --previous '91_A_FIM' \
# --batch_fisher 1000 \
# -s '91_B_FIM' \
# -d 50





