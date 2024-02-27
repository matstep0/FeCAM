#!/bin/bash

echo `date` gmum_watch: Starting initial synchronization
rsync -urltv --exclude '.git/' --progress -e ssh ./ z1165703@gmum-gw:/home/z1165703/FeCAM
echo `date` gmum_watch: Change successfully synchronized

echo `date` gmum_watch: Starting to listen on changes
fswatch -o -l 2 . | while read num;
do
    echo `date` gmum_watch: Change detected, synchronizing with server
    rsync -urltv --exclude '.git/' --progress -e ssh ./ z1165703@gmum-gw:/home/z1165703/FeCAM
    echo `date` gmum_watch: Change successfully synchronized
done