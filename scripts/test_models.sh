#!/bin/bash

TESTS="test_visual_bert test_simple_mlp_image test_simple_image"
dvc pull -f $TESTS
dvc repro $TESTS