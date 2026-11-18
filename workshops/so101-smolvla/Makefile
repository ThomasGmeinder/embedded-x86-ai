# Random VLA regression. BACKEND is mujoco or sapien.
BACKEND ?= mujoco
COUNT ?= 100

.PHONY: random sapien-setup

random:
	./scripts/run_random_regression.sh $(COUNT) --backend $(BACKEND) $(if $(STEPS),--steps $(STEPS),)

sapien-setup:
	./scripts/setup_sapien_env.sh
