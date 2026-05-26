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


## Used parameters for training

### Monte Carlo

#### Empty room

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=monte_carlo --env=empty-room --episodes=2000 --epsilon=0.8
```

#### Cliff

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=monte_carlo --env=cliff-minihack --episodes=10000 --epsilon=0.5 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=8000
```

#### Monster

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=monte_carlo --env=room-with-monster --episodes=10000 --epsilon=0.5 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=8000
```

### Sarsa

#### Empty room

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=sarsa --env=empty-room --episodes=5000 --alpha=0.25 --epsilon=0.8
```

#### Cliff

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=sarsa --env=cliff-minihack --episodes=10000 --alpha=0.05 --epsilon=0.6 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=8000
```

#### Monster

Both of these pass:

```bash
# docker compose run --rm py python train/train_tabular.py \
# --algorithm=sarsa --env=room-with-monster --episodes=15000 --alpha=0.25 --epsilon=0.8

# not very consistent passing
docker compose run --rm py python train/train_tabular.py \
--algorithm=sarsa --env=room-with-monster --episodes=15000 --alpha=0.5 --epsilon=0.5

# a bit more consistent
docker compose run --rm py python train/train_tabular.py \
--algorithm=sarsa --env=room-with-monster --episodes=10000 --alpha=0.01 --epsilon=0.6 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=8000

# very consistent
docker compose run --rm py python train/train_tabular.py \
--algorithm=sarsa --env=room-with-monster --episodes=15000 --alpha=0.01 --epsilon=0.6 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=12000
```

### Q-Learning

#### Empty room

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=q_learning --env=empty-room --episodes=5000 --alpha=0.25 --epsilon=0.8
```

#### Cliff

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=q_learning --env=cliff-minihack --episodes=15000 --alpha=0.05 --epsilon=0.6 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=12000
```

#### Monster

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=q_learning --env=room-with-monster --episodes=15000 --alpha=0.01 --epsilon=0.6 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=12000
```