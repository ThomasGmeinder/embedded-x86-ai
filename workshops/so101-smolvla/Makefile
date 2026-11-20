# Random VLA regression. BACKEND is mujoco or sapien.
BACKEND ?= mujoco
COUNT ?= 100
TASK ?= stack2-training

.PHONY: random random-stacking sapien-setup

random:
	./scripts/run_random_regression.sh $(COUNT) --backend $(BACKEND) $(if $(STEPS),--steps $(STEPS),)

random-stacking:
	./scripts/run_random_stacking_regression.sh --count $(COUNT) --backend $(BACKEND) --task $(TASK) $(if $(STEPS),--steps $(STEPS),)

sapien-setup:
	./scripts/setup_sapien_env.sh
