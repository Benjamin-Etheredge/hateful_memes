#!/bin/bash

dvc pull -f --glob test-*
dvc repro --glob test-*
