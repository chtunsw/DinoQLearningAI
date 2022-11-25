# Dino Q-Learning Bot

## Problems

- Some tutorials suggest using a sequence of frames as input. But that requires a `highly stable frame rate` in both `training` and `testing` process. If the frame rate is unstable, introducing frame sequence as input would make the performance worse. Because samples of training data would have different frame rates from each other and maybe different from the frame rate in the testing process as well.
