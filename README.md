# A3C
A3C algorithm implemented to solve various tasks in OpenAI Gym

##### Usage
```python train.py --env <ENVIRONMENT_NAME> ```

##### Files
- [```agent/network.py```](agent/network.py) Defines structure for actor and critic networks
- [```agent/worker.py```](agent/worker.py) Runs training procedure on environment
- [```utils/helper.py```](utils/worker.py) Various helper functions
- [```test.py```](test.py) Evaluates model on environment
- [```train.py```](train.py) Trains model on environment

##### Limitations
- Gets caught in suboptimal strategy for Breakout

##### Resources
- [A3C implementation](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
