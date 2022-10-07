# Valarauka
<div align = 'center'>

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](./LICENSE)
[![](https://img.shields.io/badge/python-3.5+-yellow.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/docker-latest-blue.svg)](https://www.docker.com/) 
[![](https://img.shields.io/github/stars/Mustard404/Savior.svg?label=Stars&style=social?style=plastic)](https://github.com/Mustard404/Savior/) 
[![](https://img.shields.io/github/issues/Mustard404/Savior.svg)](https://github.com/Mustard404/Savior/)
</div>

## Background
![](results/test.png)

## Frame Work
## Main Function
 This repository provides a model that supports unsupervised image segmentation and object centric concept learning. And based on these representations the knowledge-base, it is capable of learning to answer simple questions via neuro-symbolic programs. The next update of this repository will be about how to learn to generate representation and answer questions on a continuous video domain.
# Model
 The model contains mainly 3 models: 1) the scene-graph parser 2) query-to-program parser 3) neuro-symbolic knowlege base.

# Prerequisites
- torch
- torchvision
- torch_geometric
- melkor_engine
- melkor_knowledge

# Experiments
the experiment is currently performed on three different domains
- sprite qa, the domain that contains 
- battlecode world
- ptr qa dataset

## Battlecode World
in this problem setting, the world consists of 7 different kinds of objects and they are scattered across the map.

![](results/namo.jpeg)


## Sprite Question Answer
In this problem, the visual dataset contains several obejects defined as simple geometric shapes.
![](results/sprite_parse.png)

