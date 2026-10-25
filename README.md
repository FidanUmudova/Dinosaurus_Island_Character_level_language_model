# Deep Learning Specialization: Character-Level Language Modeling (Dinosaur Island)

This repository contains the implementation of a character-level Recurrent Neural Network (RNN) built from scratch using Python and NumPy. The model is trained on a dataset of dinosaur names to learn naming patterns and generate brand-new, realistic-sounding dinosaur names. These assignments are part of the **Sequence Models** course in the Deep Learning Specialization (Coursera).

---

##  Project Structure & Exercises

The project covers key components of building, optimizing, and sampling from an RNN language model:

1. **`clip` (Exercise 1):** Implements gradient clipping to prevent the exploding gradient problem during backpropagation by constraining gradient values within a specified range (`-maxValue` to `maxValue`).
2. **`sample` (Exercise 2):** Implements character sampling from the RNN's probability distribution using `np.random.choice` to generate new sequences character by character until a newline (`\n`) token is reached.
3. **`optimize` (Exercise 3):** Executes a single optimization step of stochastic gradient descent, combining forward propagation, backward propagation, gradient clipping, and parameter updates.
4. **`model` (Exercise 4):** Implements the full training loop, handling data shuffling, index cycling, input/label sequence formatting (`X` and `Y`), and periodic name generation to track training progress.

---

