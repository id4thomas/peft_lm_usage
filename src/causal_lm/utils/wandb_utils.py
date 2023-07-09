import wandb
import os

def wandb_set(project_name, config_dir, run_name):
	run = wandb_init_run(project_name, run_name)

	# Add config artifact
	config_artifact = wandb_add_artifact(
		artifact_name = "config",
		artifact_type = "config",
		artifact_files = [config_dir]
	)
	run.log_artifact(config_artifact)

def wandb_init_run(project_name, run_name, config = None):
	run = wandb.init(
		project=project_name, 
		name = run_name,
		config = config
	)
	return run

def wandb_add_artifact(artifact_name, artifact_type, artifact_files):
	artifact = wandb.Artifact(
		artifact_name,
		type = artifact_type
	)
	for artifact_file in artifact_files:
		artifact.add_file(artifact_file)
	return artifact

def wandb_watch_model(model, log_freq = 10):
	wandb.watch(
		model, 
		log="all", 
		log_freq = log_freq
	)