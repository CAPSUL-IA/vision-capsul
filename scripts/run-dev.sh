#!/bin/bash

./scripts/stop.sh

docker compose -f environments/development/docker-compose.yml up --build

./scripts/stop.sh

