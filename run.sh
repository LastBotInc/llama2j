#!/bin/bash

java -Xmx16g -XX:+UnlockExperimentalVMOptions -XX:+UseZGC --enable-preview -jar target/llama2j-1.0-SNAPSHOT.jar  "$@"
