#!/bin/bash


# # szenario 9-1
python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1000 -s 91_A_bs1000_lam10000000 -d 50
python3 src/main.py --classes 9 --learnrate 0.00001 -lambda 1000000 --iterations 2500 --batch 100 --previous 91_A_bs1000_lam10000000 --batch_fisher 1000 -s 91_B_bs1000_lam10000000 -d 50

# szenario 5-5
python3 src/main.py --classes 0 1 2 3 4  --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1000 -s 55_A_bs1000_lam10000000 -d 50
python3 src/main.py --classes 5 6 7 8 9 --learnrate 0.001 -lambda 1000000 --iterations 2500 --batch 100 --previous 55_A_bs1000_lam10000000 --batch_fisher 1000 -s 55_B_bs1000_lam10000000 -d 50

szenrio permuted
python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 9 --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1000 -s 'PM_A_bs1000_lam10000000' -d 50
python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 9 --permute 0 --learnrate 0.00001 -lambda 1000000 --iterations 25000 --batch 100 --previous 'PM_A_bs1000_lam10000000' --batch_fisher 10000 -s 'PM_B_bs1000_lam10000000' -d 50





# DEFAULT

# # szenario 9-1
# python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1000 -s 91_A -d 50
# python3 src/main.py --classes 9 --learnrate 0.00001 --iterations 2500 --batch 100 --previous 91_A --batch_fisher 1000 -s 91_B -d 50

# # szenario 5-5
# python3 src/main.py --classes 0 1 2 3 4  --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1000 -s 55_A -d 50
# python3 src/main.py --classes 5 6 7 8 9 --learnrate 0.001 --iterations 2500 --batch 100 --previous 55_A --batch_fisher 1000 -s 55_B -d 50

# szenrio permuted
# python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 9 --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1000 -s 'PM_A_BS1000' -d 50
# python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 9 --permute 0 --learnrate 0.00001 --iterations 25000 --batch 100 --previous 'PM_A_BS1000' --batch_fisher 1000 -s 'PM_B_BS1000' -d 50



# # EWC ORIGINAL

# # szenario 9-1
# python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1 -s EWC_91_A -d 50
# python3 src/main.py --classes 9 --learnrate 0.00001 --iterations 2500 --batch 100 --previous EWC_91_A --batch_fisher 1000 -s EWC_91_B -d 50

# # szenario 5-5
# python3 src/main.py --classes 0 1 2 3 4  --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1 -s EWC_55_A -d 50
# python3 src/main.py --classes 5 6 7 8 9 --learnrate 0.00001 --iterations 2500 --batch 100 --previous EWC_55_A --batch_fisher 1 -s EWC_55_B -d 50

# szenrio permuted
# python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 9 --learnrate 0.001 --iterations 2500 --batch 100 --batch_fisher 1 -s 'PM_A_BS1' -d 50
# python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 9 --permute 0 --learnrate 0.00001 --iterations 25000 --batch 100 --previous 'PM_A_BS1' --batch_fisher 1 -s 'PM_B_BS1' -d 50
