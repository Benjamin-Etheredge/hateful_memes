#!/bin/bash

# dvc pull -f --glob test-*
dvc repro -s --glob test_*
