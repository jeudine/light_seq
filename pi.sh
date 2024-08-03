#!/bin/bash
cd "$(dirname "$0")"
git pull
cargo run -r -- 0 &&
sudo shutdown now
