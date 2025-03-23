# 🧠 2048 AI - Monte Carlo CPU + GPU (CUDA + Numba)

Ce projet implémente une Intelligence Artificielle pour jouer automatiquement au jeu **2048**, en utilisant la stratégie de **Monte Carlo Tree Search simplifiée** (random rollouts), avec deux versions :

- ✅ Version **CPU** (`code_cpu.py`)
- ⚡ Version **GPU** (`code_gpu.py`) accélérée via **Numba + CUDA**, avec multi-streams et rollouts massifs

---

## 🚀 Fonctionnalités

- 📊 **Évaluation de coups** via simulations aléatoires (rollouts)
- 🧠 Choix du meilleur coup par moyenne des scores finaux simulés
- 💻 Version **CPU** simple avec `NumPy` + `random`
- 🔥 Version **GPU** hautes performances avec `Numba.cuda`
- 👀 Affichage **live** de l'avancement (step, score, temps, coups possibles)
- 💾 Optimisé pour les GPU NVIDIA compatibles CUDA

---

## 🧩 Structure du projet

```
.
├── code_cpu.py     # Version CPU de l'IA
├── code_gpu.py     # Version GPU avec CUDA
├── README.md
```

---

## ⚙️ Dépendances

Installe les dépendances nécessaires :

```bash
pip install numpy numba
```

⚠️ Pour utiliser la version GPU :
- Tu dois avoir une **carte NVIDIA** compatible CUDA
- Le toolkit **CUDA** doit être installé et reconnu par `numba`

---

## 🧪 Exécution

### Version CPU :

```bash
python code_cpu.py
```

### Version GPU :

```bash
python code_gpu.py
```

---

## 📈 Exemple de sortie

```
[1/500] Move: LEFT, Score: 24, Moves left: ['UP', 'DOWN', 'RIGHT'], Time: 0.28s
[2/500] Move: UP, Score: 40, Moves left: ['LEFT', 'RIGHT'], Time: 0.59s
...
Partie terminée !
[[   2    4   16   32]
 [  64   32   16    2]
 [   2   16   64   32]
 [   0    0    2    0]]
Score final = 382
Temps écoulé : 4.73 s
```

---

## 📌 Détails techniques

### CPU

- Nombre de rollouts : configurable (`n_rollouts`)
- Nombre max d'étapes : configurable (`n_steps`)
- Affichage du score et des mouvements possibles à chaque étape

### GPU

- `rollouts` massifs (jusqu'à 8192 ou plus)
- `numba.cuda.jit` avec `local arrays`, logique fusionnée manuellement
- Optimisé pour les rollouts simultanés via **multi-streams**

---

## 📚 Inspirations & Objectifs

Projet développé pour explorer :

- Les techniques de **Monte Carlo Simulation**
- L’optimisation d’algorithmes sur GPU avec **Numba CUDA**
- La portabilité et la performance en **IA de jeux simples**

---

## ✨ Auteur

👤 **Maxence Gomes**  
💼 Ingénieur logiciel, passionné par l’IA, la virtualisation et le trading quantitatif.

- [LinkedIn](https://www.linkedin.com/in/maxence-gomes-714283165/)
- [GitHub](https://github.com/supergmax)

---

## 📜 Licence

Ce projet est open-source sous licence **MIT**.

---

> "Make your AI smarter, your GPU hotter, and your code cleaner." — Tonton Max 😎
