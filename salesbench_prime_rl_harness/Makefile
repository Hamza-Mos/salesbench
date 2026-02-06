.DEFAULT_GOAL := help
.PHONY: help setup show-config smoke eval benchmark benchmark-tiny tui rl

# Source-of-truth defaults copied from original harness:
#   - salesbench/runner/config.py MODE_PRESETS
#   - salesbench/models.py DEFAULT_BENCHMARK_MODELS
#
# Legacy preset notes from original harness:
#   production: episodes=100, leads=100, total_hours=80, parallelism=1, safety_max_turns=None
#   demo:       episodes=3,   leads=50,  total_hours=2,  parallelism=1, safety_max_turns=None
#   test:       episodes=3,   leads=5,   total_hours=16, parallelism=1, safety_max_turns=None
#   debug:      episodes=1,   leads=5,   total_hours=4,  parallelism=1, safety_max_turns=None
#
# Important: original CLI help text says "demo (10 eps)", but MODE_PRESETS actually uses 3.
# This Makefile follows MODE_PRESETS (the actual runtime source of truth).

ENV ?= salesbench-prime-rl
MODE ?= production

# Original DEFAULT_BENCHMARK_MODELS from salesbench/models.py
MODEL ?= openai/gpt-5.2
MODELS ?= openai/gpt-5.2 anthropic/claude-opus-4-5-20251101 google/gemini-3-pro-preview

ROLLOUTS ?= 1
SPLIT ?= eval
SEED ?= 42
EXTRA_FLAGS ?=
UPLOAD_FLAGS ?= --skip-upload

VALID_MODES := production demo test debug
ifeq (,$(filter $(MODE),$(VALID_MODES)))
$(error MODE must be one of: $(VALID_MODES))
endif

# Mode-specific defaults (mapped from original total_hours to work_days x hours_per_day)
ifeq ($(MODE),production)
EPISODES ?= 100
NUM_LEADS ?= 100
WORK_DAYS ?= 10
HOURS_PER_DAY ?= 8
CONCURRENCY ?= 1
MAX_TURNS ?=
endif

ifeq ($(MODE),demo)
EPISODES ?= 3
NUM_LEADS ?= 50
WORK_DAYS ?= 1
HOURS_PER_DAY ?= 2
CONCURRENCY ?= 1
MAX_TURNS ?=
endif

ifeq ($(MODE),test)
EPISODES ?= 3
NUM_LEADS ?= 5
WORK_DAYS ?= 2
HOURS_PER_DAY ?= 8
CONCURRENCY ?= 1
MAX_TURNS ?=
endif

ifeq ($(MODE),debug)
EPISODES ?= 1
NUM_LEADS ?= 5
WORK_DAYS ?= 1
HOURS_PER_DAY ?= 4
CONCURRENCY ?= 1
MAX_TURNS ?=
endif

help:
	@echo "Targets:"
	@echo "  make setup"
	@echo "  make show-config"
	@echo "  make smoke MODE=debug"
	@echo "  make eval MODE=production MODEL=openai/gpt-5.2"
	@echo "  make benchmark MODE=production"
	@echo "  make benchmark-tiny  # quick end-to-end sweep (~few minutes)"
	@echo "  make tui"
	@echo "  make rl"
	@echo ""
	@echo "Common overrides:"
	@echo "  MODE={production|demo|test|debug}"
	@echo "  MODEL=<provider/model>"
	@echo "  MODELS='<provider/model> <provider/model> ...'"
	@echo "  EPISODES, CONCURRENCY, NUM_LEADS, WORK_DAYS, HOURS_PER_DAY"
	@echo "  MAX_TURNS=<int>  # optional safety cap; blank keeps legacy None behavior"
	@echo "  EXTRA_FLAGS='-k OPENAI_API_KEY -b https://api.openai.com/v1'"
	@echo ""
	@echo "Defaults from MODE=$(MODE):"
	@echo "  EPISODES=$(EPISODES) CONCURRENCY=$(CONCURRENCY) NUM_LEADS=$(NUM_LEADS)"
	@echo "  WORK_DAYS=$(WORK_DAYS) HOURS_PER_DAY=$(HOURS_PER_DAY) SEED=$(SEED) SPLIT=$(SPLIT)"
	@if [ -n "$(MAX_TURNS)" ]; then \
		echo "  MAX_TURNS=$(MAX_TURNS)"; \
	else \
		echo "  MAX_TURNS=<legacy-none> (omitted from env args)"; \
	fi

setup:
	uv sync
	uv pip install -e .

show-config:
	@$(MAKE) help

smoke:
	$(MAKE) eval MODE=debug MODEL=$(MODEL) EPISODES=1 CONCURRENCY=1 NUM_LEADS=5 WORK_DAYS=1 HOURS_PER_DAY=4

eval:
	@env_args=$$( \
		json=$$(printf '{"split":"%s","seed":%s,"num_leads":%s,"work_days":%s,"hours_per_day":%s,"num_examples":%s,"eval_num_examples":%s' \
			"$(SPLIT)" "$(SEED)" "$(NUM_LEADS)" "$(WORK_DAYS)" "$(HOURS_PER_DAY)" "$(EPISODES)" "$(EPISODES)"); \
		if [ -n "$(MAX_TURNS)" ]; then json="$$json,\"max_turns\":$(MAX_TURNS)"; fi; \
		printf "%s}" "$$json" \
	); \
	prime eval run $(ENV) -m $(MODEL) -n $(EPISODES) -r $(ROLLOUTS) -c $(CONCURRENCY) -a "$$env_args" $(EXTRA_FLAGS) $(UPLOAD_FLAGS)

benchmark:
	@env_args=$$( \
		json=$$(printf '{"split":"%s","seed":%s,"num_leads":%s,"work_days":%s,"hours_per_day":%s,"num_examples":%s,"eval_num_examples":%s' \
			"$(SPLIT)" "$(SEED)" "$(NUM_LEADS)" "$(WORK_DAYS)" "$(HOURS_PER_DAY)" "$(EPISODES)" "$(EPISODES)"); \
		if [ -n "$(MAX_TURNS)" ]; then json="$$json,\"max_turns\":$(MAX_TURNS)"; fi; \
		printf "%s}" "$$json" \
	); \
	for m in $(MODELS); do \
		prime eval run $(ENV) -m "$$m" -n $(EPISODES) -r $(ROLLOUTS) -c $(CONCURRENCY) -a "$$env_args" $(EXTRA_FLAGS) $(UPLOAD_FLAGS); \
	done

benchmark-tiny:
	$(MAKE) benchmark MODE=demo EPISODES=2 CONCURRENCY=1 MODELS="openai/gpt-4.1-mini openai/gpt-4.1" MAX_TURNS=25

tui:
	prime eval tui

rl:
	prime rl run configs/lab/salesbench-prime-rl.toml
