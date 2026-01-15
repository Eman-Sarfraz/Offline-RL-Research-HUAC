
# HUAC: Hybrid Uncertainty-Aware Adaptive Conservatism for Offline RL

HUAC is a novel Offline Reinforcement Learning framework designed for safety-critical domains, such as Finance and Portfolio Optimization. It addresses the fundamental trade-off between performance and conservatism by dynamically adjusting pessimistic penalties based on **Epistemic Uncertainty**.

## 🚀 Key Features

* **Adaptive Conservatism:** Automatically scales Q-value penalties using variance from a Deep Ensemble.
* **Implicit Value Learning:** Leverages asymmetric expectile regression (inspired by IQL) for stable convergence.
* **Portfolio Optimization Environment:** Includes a custom OpenAI Gym-compatible environment for financial asset management.
* **Uncertainty-Aware:** Explicitly quantifies "what the model doesn't know" to prevent risky actions in out-of-distribution (OOD) states.

## 🛠 Project Structure

* `huac_rl.py`: The core implementation of the HUAC algorithm, including the Actor, Value, and Ensemble Critic networks.
* `finance_env.py`: A `gym.Env` implementation for portfolio optimization with synthetic financial data generation.
* `train_eval.py`: The main entry point for training the model and evaluating it against behavioral baselines.
* `methodology.md`: Detailed research background, mathematical formulation, and ablation study plans.

## 🧬 Methodology

HUAC operates on three primary pillars:

1. **Ensemble Estimation:** Uses multiple Critic networks to estimate Q-values.
2. **Variance-Based Regularization:** Higher variance between Critics triggers stronger conservatism.
3. **Stability:** Uses an Implicit Value function to provide a baseline for advantage estimation without interacting with the environment.

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/HUAC-RL.git
cd HUAC-RL

```


2. Install dependencies:
```bash
pip install torch numpy gymnasium matplotlib tqdm

```



## 📈 Usage

### Training and Evaluation

To collect offline data, train the HUAC policy, and generate a performance plot:

```bash
python train_eval.py

```

The script will:

1. Generate synthetic financial data.
2. Collect a "medium-quality" offline dataset.
3. Train the HUAC agent for 10,000 iterations.
4. Save a result plot to `results_plot.png`.

### Customizing the Algorithm

You can adjust the hyper-parameters in `huac_rl.py` or `train_eval.py`:

* `tau`: Target network update rate.
* `expectile`: The asymmetry parameter for the Value function (default `0.7`).
* `ensemble_size`: Number of Critics (default `5`).

## 📊 Performance Visualization

After running the training script, the model compares the learned HUAC policy against the behavioral baseline (the policy used to collect the initial data).

## 📝 Citation


> Yasin, A., Sarfraz, E., & Rehman, A. (2026). *Advancing Data-Driven Policy Learning with Optimized Offline Reinforcement Learning: A Comprehensive Framework for Uncertainty-Aware Regularization with Adaptive Conservatism.*

---

*Developed at the University of Central Punjab, Lahore.*
