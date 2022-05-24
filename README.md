# LocalGuard paper
This demo is implementation for the paper "LocalGuard: Guard the Vertical Federated Graph Learning from Property Inference Attack" (TNSE).

In this paper, we propose a perturbation-based defense method, LocalGuard, which balances the privacy and accuracy. By introducing the noise generator module in the LocalGuard, the entropy between noisy embeddings and private properties is increased to protect privacy.


# How to run  

## PiAttack

Take GCN on Cora dataset as example, which is running on default arguments.

```python
cd /localguard/PiAttack
```

Run PiAttack on Cora dataset.

```python
python attack_cora.py
```

Experiment intermediate results can be modified in /localguard/tmp

## LocalGuard

Take GCN on Cora dataset as example, which is running on default arguments.

```python
cd /localguard/LocalGuard
```

Run LocalGuard on Cora dataset.

```python
python defense_cora.py
```

Experiment intermediate results can be modified in /localguard/tmp

# Requirements
Python >= 3.6
Pytorch >= 1.1.0

# License and Copyright
The project is open source under MIT license.
