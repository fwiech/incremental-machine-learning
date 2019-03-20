#setup for permuted mnist!
#python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 9 --learnrate 0.001 --iterations 3000 --batch 100 -s FISH1 --batch_fisher 1000 -d 100
#python3 src/main.py --classes 0 1 2 3 4 5 6 7 8 9 --learnrate 0.001 --lambda 1000 --iterations 2500 --batch 100 -s FISH2 --previous FISH1 --batch_fisher 1 -d 100 --permute 0

#setup for 9-1!
python3 src/main.py --classes 0 1 2 3 4 5 6 7 8  --learnrate 0.001 --iterations 2500 --batch 100 -s FISH1 --batch_fisher 1000 -d 100
python3 src/main.py --classes 9 --learnrate 0.00001 --lambda 1000000 --iterations 2500 --batch 100  --previous FISH1 --batch_fisher 1 -d 100

### test testing arrangements!!
### check whether fm is loaded correctly!!


