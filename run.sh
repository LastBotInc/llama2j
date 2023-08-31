#!/bin/bash

java -Xmx32g -XX:+UnlockExperimentalVMOptions -XX:+UseZGC --enable-preview -jar target/llama2j-1.0-SNAPSHOT.jar  "$@"
