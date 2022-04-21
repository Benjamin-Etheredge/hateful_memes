#!/bin/bash

# dvc pull -f --glob test-*
dvc repro -f -s --glob test_*
