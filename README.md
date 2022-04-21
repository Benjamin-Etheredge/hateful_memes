# hateful_memes
[![Training](https://github.com/Benjamin-Etheredge/hateful_memes/actions/workflows/train.yml/badge.svg?branch=master)](https://github.com/Benjamin-Etheredge/hateful_memes/actions/workflows/train.yml)

# Random Info



The project syncs with a remote s3 bucket (minio) to manage data, trained models, and metrics. Therefore, you have the credentials in your environment variables to pull the files. An example of the expected environment variables can be found in `.env.sample`. 

Launching the project through VSCode devcontainers is the simpliest way to get up and running. You still have to add your own `.env` file with your credentials, but the rest should be handled for you.


`dvc.yaml` is bascially a make file. It lays out how to reproduce results. It has stages that list commands, dependencies, and outputs. 
Named stages can be reprodced with `dvc repro {stage}`.

All stages can be run with `dvc repro`, but I do not recommend it. 

After a `repro`, DVC caches the results and updates `dvc.lock` with the hashes of the files. This is how it tracks the files. You probably won't have to check that in as CI/CD will update the file when it reproduces things. 

`./scripts/test_models.sh` should reproduce all testing stages and can be run locally to test changes to existing stages. If you add a test stage, you can add it here. Or you can also just repro it yourself. 
