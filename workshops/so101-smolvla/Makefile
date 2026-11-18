# Random VLA regression. From this directory: make random
# COUNT=10 runs ten seeds. Omit COUNT for the default of 100.
.PHONY: random sapien sapien-setup

random:
	BACKEND=$(BACKEND) ./scripts/run_random_regression.sh $(COUNT)

sapien:
	BACKEND=sapien ./scripts/run_random_regression.sh $(COUNT)

sapien-setup:
	./scripts/setup_sapien_env.sh
