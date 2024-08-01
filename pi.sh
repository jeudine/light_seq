#!/bin/bash
cd "$(dirname "$0")"
git pull
cargo run -r
sudo shutdown now
