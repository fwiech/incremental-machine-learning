.PHONY: D55 D55_FIM_A D55_FIM_B D55_GM_A D55_GM_B plot_D55 plot_D55_FIM plot_D55_GM

checkpoint_55_A_FIM := '55_FIM_A'
checkpoint_55_B_FIM := '55_FIM_B'

checkpoint_55_A_GM := '55_GM_A'
checkpoint_55_B_GM := '55_GM_B'

D55: D55_FIM_A D55_FIM_B D55_GM_A D55_GM_B plot_D55_FIM plot_D55_GM

plot_D55: plot_D55_FIM plot_D55_GM

D55_FIM_A:
	@echo "train first task with FIM"
	python3 src/main.py \
	--classes 0 1 2 3 4 \
	--learnrate 0.001 \
	--iterations 2500 \
	--batch_train 100 \
	--batch_matrix 1 \
	-s ${checkpoint_55_A_FIM} \
	-d 50

D55_FIM_B:
	@echo "Task T2 on FIM"
	python3 src/main.py \
	--classes 5 6 7 8 9 \
	--learnrate 0.00001 \
	--iterations 2500 \
	--batch_train 100 \
	--previous ${checkpoint_55_A_FIM} \
	--batch_matrix 1 \
	-s ${checkpoint_55_B_FIM} \
	-d 50

D55_GM_A:
	@echo "train first task with GM (BS1000)"
	python3 src/main.py \
	--classes 0 1 2 3 4 \
	--learnrate 0.001 \
	--iterations 2500 \
	--batch_train 100 \
	--batch_matrix 1000 \
	-s ${checkpoint_55_A_GM} \
	-d 50

D55_GM_B:
	@echo "Task T2 on GM"
	python3 src/main.py \
	--classes 5 6 7 8 9 \
	--learnrate 0.00001 \
	--iterations 2500 \
	--batch_train 100 \
	--previous ${checkpoint_55_A_GM} \
	--batch_matrix 1000 \
	-s ${checkpoint_55_B_GM} \
	-d 50

plot_D55_FIM:
	python3 src/plots/training.py \
	--title 'D5-5 FIM' \
	-t1 checkpoints/${checkpoint_55_A_FIM}/ \
	-t2 checkpoints/${checkpoint_55_B_FIM}/ \
	-s D55_FIM

plot_D55_GM:
	python3 src/plots/training.py \
	--title 'D5-5 GM' \
	-t1 checkpoints/${checkpoint_55_A_GM}/ \
	-t2 checkpoints/${checkpoint_55_B_GM}/ \
	-s D55_GM