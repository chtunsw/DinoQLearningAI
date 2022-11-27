# Dino Q-Learning Bot

## Description

This project builds a game bot for Chrome Dino Game using q-learning algorithm.

<p align="center">
  <img src="assets/gameplay.gif" width="300">
</p>

## Performance

After training for 10,000 episodes (16 hours on a 2017 Macbook Pro), with the best model chosen from the validation process, the bot can get a stable 100+ score in most of games. The highest score this model got during the test was 834.

<p align="center">
  <img src="assets/train_history.png" align="center" width="400">
</p>

## Environment

- Device: 2017 Macbook Pro
- Python: 3.7.7
- Dependency: see [requirements.txt](requirements.txt)

## Instructions

Train and save models:

```python
python app.py train
```

Validate trained models:

```python
python app.py validate
```

Test with trained model:

```python
python app.py test
```

## Some thoughts

- Frame rate plays an important role in this project and it varies on different devices. The delay between `frame` and `next_frame` (state returned from the environment) should be short enough so that the model can get a real-time feedback.
- Some tutorials suggest using a sequence of frames as input. But that requires a highly stable frame rate in both `training` and `testing` process. If the frame rate is unstable, introducing frame sequence as input would make the performance worse. Because samples of training data would have different frame rate from each other and the frame rate in the test process.
- At the end of the training process, the performance was still at a rising trend. Maybe the model can get a better performance given more training time.

## Possible improvements

- Train for more episodes. (Maybe 100,000)
- Remove unnecessary debug code (logs, display frame) to increase frame rate.
- Train and save model in another process, so that it won't interrupt the sample process.
- Save the model in time when the bot makes a new record in training process.
- Try use more advanced variants of q-learning algorithm.
