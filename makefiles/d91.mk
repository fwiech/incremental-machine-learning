.PHONY: D91 D91_FIM_A D91_FIM_B D91_GM_A D91_GM_B

checkpoint_91_A_FIM := '91_A_FIM'
checkpoint_91_B_FIM := '91_B_FIM'

checkpoint_91_A_GM := '91_A_GM'
checkpoint_91_B_GM := '91_B_GM'

D91: D91_FIM_A D91_FIM_B D91_GM_A D91_GM_B

D91_FIM_A:
	@echo "task T1 with FIM"
	python3 src/main.py \
	--classes 0 1 2 3 4 5 6 7 8 \
	--learnrate 0.001 \
	--iterations 2500 \
	--batch_train 100 \
	--batch_matrix 1 \
	-s ${checkpoint_91_A_FIM} \
	-d 50

D91_FIM_B:
	@echo "Task T2 on FIM"
	python3 src/main.py \
	--classes 9 \
	--learnrate 0.00001 \
	--iterations 2500 \
	--batch_train 100 \
	--previous ${checkpoint_91_A_FIM} \
	--batch_matrix 1 \
	-s ${checkpoint_91_B_FIM} \
	-d 50

D91_GM_A:
	@echo "task T1 with GM (BS1000)"
	python3 src/main.py \
	--classes 0 1 2 3 4 5 6 7 8 \
	--learnrate 0.001 \
	--iterations 2500 \
	--batch_train 100 \
	--batch_matrix 1000 \
	-s ${checkpoint_91_A_GM} \
	-d 50

D91_GM_B:
	@echo "Task T2 on GM (BS1000)"
	python3 src/main.py \
	--classes 9 \
	--learnrate 0.00001 \
	--iterations 2500 \
	--batch_train 100 \
	--previous ${checkpoint_91_A_GM} \
	--batch_matrix 1000 \
	-s ${checkpoint_91_B_GM} \
	-d 50
