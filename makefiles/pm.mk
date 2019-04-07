.PHONY: PM PM_FIM_A PM_FIM_B PM_GM_A PM_GM_B

checkpoint_PM_A_FIM := 'PM_A_FIM'
checkpoint_PM_B_FIM := 'PM_B_FIM'

checkpoint_PM_A_GM := 'PM_A_GM'
checkpoint_PM_B_GM := 'PM_B_GM'

PM: PM_FIM_A PM_FIM_B PM_GM_A PM_GM_B

PM_FIM_A:
	@echo "task T1 with FIM"
	python3 src/main.py \
	--classes 0 1 2 3 4 5 6 7 8 9 \
	--learnrate 0.001 \
	--iterations 2500 \
	--batch_train 100 \
	--batch_matrix 1 \
	-s ${checkpoint_PM_A_FIM} \
	-d 50

PM_FIM_B:
	@echo "Task T2 on FIM"
	python3 src/main.py \
	--classes 0 1 2 3 4 5 6 7 8 9 \
	--learnrate 0.00001 \
	--iterations 20000 \
	--batch_train 100 \
	--permute 0 \
	--previous ${checkpoint_PM_A_FIM} \
	--batch_matrix 1 \
	-s ${checkpoint_PM_B_FIM} \
	-d 1000

PM_GM_A:
	@echo "task T1 with GM (BS1000)"
	python3 src/main.py \
	--classes 0 1 2 3 4 5 6 7 8 9 \
	--learnrate 0.001 \
	--iterations 2500 \
	--batch_train 100 \
	--batch_matrix 1000 \
	-s ${checkpoint_PM_A_GM} \
	-d 50

PM_GM_B:
	@echo "Task T2 on GM (BS1000)"
	python3 src/main.py \
	--classes 0 1 2 3 4 5 6 7 8 9 \
	--learnrate 0.00001 \
	--iterations 20000 \
	--batch_train 100 \
	--permute 0 \
	--previous ${checkpoint_PM_A_GM} \
	--batch_matrix 1000 \
	-s ${checkpoint_PM_B_GM} \
	-d 1000
