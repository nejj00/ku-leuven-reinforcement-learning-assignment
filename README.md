# rl-assignment-26-private



## Getting started
This project is based on Docker. If you are not familiar with Docker, you can find more information and installation instructions on the [Docker website](https://www.docker.com/).

In the root of the repo:
    
    docker compose build

 Notice that the current Dockerfile installs a CPU-only version of PyTorch. 

To run the test script:

    docker compose run --rm py pytest tests/test1.py

For any other script, you can run:

    docker compose run --rm py python <script_name.py>
