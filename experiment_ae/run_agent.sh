#!/bin/bash


# wandb sweep sweep.yaml
SWEEP_ID=$1

for i in {1..4} 
do
	screen -S "agent_$i" -d -m wandb agent "$SWEEP_ID"
done

