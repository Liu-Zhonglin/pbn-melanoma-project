# Optimal Temporal Control of Melanoma Immunotherapy Resistance

This repository contains the complete source code and models for the paper: **"Dynamic Network Analysis and Optimal Control Reveal a ‘Hit-and-Run’ Mechanism to Reverse Immunotherapy Resistance in Melanoma"**.

This work investigates the complex gene regulatory network underlying innate resistance to anti-PD-1 immunotherapy in melanoma. We constructed a Probabilistic Boolean Network (PBN) from patient data and used Reinforcement Learning (RL) to discover novel therapeutic strategies. The central finding is the identification of a **‘hit-and-run’ mechanism**, where a brief, transient inhibition of a key gene (MAPK3 or JUN) is significantly more effective at reversing the resistant phenotype than a longer, sustained intervention.

This code provides all necessary components to replicate the findings presented in the paper.

## Key Features

* **PBN Model**: A 12-gene Probabilistic Boolean Network model of anti-PD-1 resistant melanoma, inferred from patient data (Hugo et al., 2016).
* **Reinforcement Learning Framework**: An optimal control environment built using `gym-PBN-stac` and `stable-baselines3`.
* **Optimal Control Scripts**: The repository includes Python scripts that perform the key analyses described in the paper:
    * Training a PPO agent to control the resistant network.
    * Validating static knockout strategies.
    * Systematically testing the "hit-and-run" hypothesis by comparing different intervention targets (MAPK3, JUN) and durations (1-step, 2-step).


```

## Experimental Pipeline

The primary script, `run_temporal_hypothesis_analysis.py`, encapsulates the complete experimental pipeline. When executed, it first trains a Reinforcement Learning agent (or loads a pre-trained one if available) and then uses this agent to systematically evaluate the four temporal intervention strategies detailed in the paper. Upon completion, it generates the summary (`.txt`) and results (`.csv`) files, which contain the data used to generate Table 1 in the main paper.

## Environment and Dependencies

The analysis was conducted using **Python 3.9+**. All required packages are listed in the `requirements.txt` file.



## License

This project is licensed under the MIT License.

## Acknowledgments

* This work was inspired by and uses data from the foundational study by **Hugo et al.** (Cell, 2016).
* The reinforcement learning environment was built using the **gym-PBN-stac** package.
* The PPO agent was implemented using the **stable-baselines3** library.

---


# Core packages for simulation and RL
gymnasium
stable-baselines3[extra]
gym-PBN-stac

# Standard data science packages
numpy
pandas
`
