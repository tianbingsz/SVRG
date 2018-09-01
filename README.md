## SVRG
* Contributors and Collaborators: Tianbing Xu (Baidu Research, CA), Qiang Liu
  (UT, Austin), Jian Peng (UIUC), Liang Zhao (Baidu Research, CA), Andrew Zhang
(Stanford University)

### Major Contributions: 
* We introduce the stochastic variance reduced gradient descent (SVRG) to 
model-free policy gradient to improve the sample-efficiency.
* The SVRG estimation is incorporated into a trust-region Newton 
conjugate gradient framework for the policy optimization.
* SVRG is developped based on the rllab (https://github.com/rll/rllab)

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

## Results (TODO, figures and video)

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

