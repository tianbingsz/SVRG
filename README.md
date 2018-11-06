## SVRG
* Contributors and Collaborators: Tianbing Xu (Baidu Research, CA), Qiang Liu
  (UT, Austin), Jian Peng (UIUC)

### Contributions: 
The variance of the policy gradient estimates obtained from the simulation is often excessive, leading to poor sample efficiency. In this paper, we apply the stochastic variance reduced gradient descent (SVRG) to model-free policy gradient to improve the sample-efficiency. The SVRG estimation is incorporated into a trust-region Newton conjugate gradient framework for the policy optimization.

## Dependencies
* Rllab (https://github.com/rll/rllab)
* Python 3.6
* The Usual Suspects: NumPy, matplotlib, scipy
* TensorFlow
* gym - [installation instructions](https://gym.openai.com/docs)
* [MuJoCo](http://www.mujoco.org/) (30-day trial available and free to students)

Refer to requirements.txt for more details.

### Running Command
* After launching the virtual env, set up PYTHONPATH and Mujoco PATH,
```
source start.sh
```

* Run experiment
```
cd sandbox/rocky/tf/launchers/
python trpo_gym_swimmer.py
```

## Results (MuJoco Robotics Tasks)
![hopper_svrg](https://user-images.githubusercontent.com/22249000/44967751-45bdf700-aef8-11e8-8280-252ef345ade7.jpg)
[![Half-Cheetah](https://img.youtube.com/vi/YheWgjt9eww/0.jpg)](https://www.youtube.com/watch?v=YheWgjt9eww)
[![Hopper](https://img.youtube.com/vi/9Eu8mmEskwQ/0.jpg)](https://www.youtube.com/watch?v=9Eu8mmEskwQ)
 
## Reference
* Tianbing Xu, Qiang Liu, Jian Peng, "Stochastic Variance Reduction 
for Policy Gradient Estimation", arXiv, 2017
* S. S. Du, J. Chen, L. Li, L. Xiao, and D. Zhou, “Stochastic variance
reduction methods for policy evaluation,”ICML, 2017
* R. Johnson and T. Zhang, “Accelerating stochastic gradient descent
using predictive variance reduction, NIPS 2013
* A. Owen and Y. Zhou, “Safe and effective importance sampling,”JASA, 2000
* Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel. 
"Benchmarking Deep Reinforcement Learning for Continuous Control". ICML 2016

