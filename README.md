# Enhancing Closest Point of Approach (eCPA)

Welcome to the official repository for the paper titled **"Steering Towards Maritime Safety with True Motion Predictions Ensemble"** presented at the ACSOS 2024 - Self-Improving Systems Integration Workshop.

## Overview

This repository contains the code implementation of our Just-In-Time (JIT) tool designed to enhance the performance of the Closest Point of Approach (CPA) calculations. Our approach leverages ensemble prediction methods to improve the accuracy and reliability of ship motion forecasting, thereby promoting maritime safety.

## Publication

**Title:** Steering Towards Maritime Safety with True Motion Predictions Ensemble

**Conference:** ACSOS 2024 - Self-Improving Systems Integration Workshop

## Introduction

This repository contains the code for paper titles: "Steering Towards Maritime Safety with True Motion Predictions Ensemble"

As illustrated in Figure, the main components of the pipeline include user adaptable methods of
trajectories interpolation, prediction, anomalies elimination,

![eCPA pipline](imgs/Approach.png)

## Requirements

To install all the requirements, one needs to first install:

+ conda
+ poetry

A detailed list of the required libraries can be found in:

+ poetry.toml

The proper installation must then be done with poetry and conda.

## Generated Encounter Scenarios

![Scenarios](imgs/scenarios.png)

## Contributing
![results](imgs/scenarios_prediction.png)

By considering the actual dimensions and headings
of vessels, as well as environmental factors such as wind
and current, the eCPA method provides a more accurate and
reliable assessment of collision risks.

## Contact

For any questions or inquiries, please contact the authors.
[gaf, stu222518, sgao, st]@informatik.uni-kiel.de

## License
We use the MIT license, see the `LICENSE` file for details.



## Citation
