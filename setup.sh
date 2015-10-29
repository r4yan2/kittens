#!/bin/bash

apt-get -y update
apt-get -y upgrade
apt-get install -y git byobu htop
cd /dev/shm/
git pull https://github.com/r4yan2/kittens
cd kittens
python main.py
exit 0
