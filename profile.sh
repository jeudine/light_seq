#!/bin/bash
cargo run -r --features profile
./plot.gpi profile.tsv
