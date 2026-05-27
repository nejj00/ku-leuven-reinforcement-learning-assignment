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
--algorithm=monte_carlo --env=empty-room --episodes=200 --epsilon=0.8
```

#### Cliff

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=monte_carlo --env=cliff-minihack --episodes=10000 --epsilon=0.5 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=8000
```

#### Monster

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=monte_carlo --env=room-with-monster --episodes=15000 --epsilon=0.6 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=12000
```

### Sarsa

#### Empty room

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=sarsa --env=empty-room --episodes=200 --alpha=0.25 --epsilon=0.8
```

#### Cliff

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=sarsa --env=cliff-minihack --episodes=10000 --alpha=0.1 --epsilon=0.5 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=8000
```

#### Monster

```bash
# a bit more consistent
docker compose run --rm py python train/train_tabular.py \
--algorithm=sarsa --env=room-with-monster --episodes=2000 --alpha=0.1 --epsilon=1.0 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=1000
```

### Q-Learning

#### Empty room

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=q_learning --env=empty-room --episodes=200 --alpha=0.25 --epsilon=0.8
```

#### Cliff

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=q_learning --env=cliff-minihack --episodes=10000 --alpha=0.1 --epsilon=0.5 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=8000
```

#### Monster

```bash
docker compose run --rm py python train/train_tabular.py \
--algorithm=q_learning --env=room-with-monster --episodes=2000 --alpha=0.1 --epsilon=1.0 --epsilon-schedule --epsilon-end=0.01 --epsilon-decay-episodes=1500
```

```bash
# test
docker compose run --rm py pytest -vv tests/test2.py::test_q_learning_improves_on_room_with_monster
```

### DQN


#### Empty room

```bash
# this works
docker compose run --rm py python train/train_dqn.py \
  --env=empty-room \
  --episodes=3000 \
  --learning-rate=0.001 \
  --epsilon-start=1.0 \
  --epsilon-end=0.05 \
  --epsilon-decay-steps=50000 \
  --batch-size=64 \
  --buffer-size=10000 \
  --learning-starts=500 \
  --train-frequency=4 \
  --target-network-frequency=500 \
  --tau=1.0
```

#### Cliff

```bash
# this works
docker compose run --rm py python train/train_dqn.py \
  --env=cliff-minihack \
  --episodes=10000 \
  --learning-rate=0.001 \
  --epsilon-start=1.0 \
  --epsilon-end=0.05 \
  --epsilon-decay-steps=100000 \
  --batch-size=64 \
  --buffer-size=10000 \
  --learning-starts=500 \
  --train-frequency=4 \
  --target-network-frequency=500 \
  --tau=1.0
```

#### Monster

```bash
docker compose run --rm py python train/train_dqn.py \
  --env=room-with-monster \
  --episodes=10000 \
  --learning-rate=0.001 \
  --epsilon-start=1.0 \
  --epsilon-end=0.05 \
  --epsilon-decay-steps=100000 \
  --batch-size=64 \
  --buffer-size=10000 \
  --learning-starts=500 \
  --train-frequency=4 \
  --target-network-frequency=500 \
  --tau=1.0
```

### PPO

#### Empty room 

```bash
docker compose run --rm py python train/train_ppo.py \
--env=empty-room \
--episodes=100 \
--learning-rate=0.001 \
--gae-lambda=0.95 \
--clip-coef=0.2 \
--num-steps=128 \
--num-minibatches=4 \
--update-epochs=4 \
--ent-coef=0.01 \
--vf-coef=0.5
```

#### Cliff

```bash
docker compose run --rm py python train/train_ppo.py \
--env=cliff-minihack \
--episodes=500 \
--learning-rate=0.001 \
--gae-lambda=0.97 \
--clip-coef=0.2 \
--num-steps=128 \
--num-minibatches=4 \
--update-epochs=4 \
--ent-coef=0.03 \
--vf-coef=0.5
```

#### Monster

```bash
docker compose run --rm py python train/train_ppo.py \
--env=room-with-monster \
--episodes=1000 \
--learning-rate=0.001 \
--gae-lambda=0.97 \
--clip-coef=0.2 \
--num-steps=128 \
--num-minibatches=4 \
--update-epochs=4 \
--ent-coef=0.04 \
--vf-coef=0.5
```