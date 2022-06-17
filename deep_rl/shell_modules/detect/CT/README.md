# Task Dissimilarity
- The class [dissimilarity](https://github.com/DMIU-ShELL/deeprl-shell/blob/main/deep_rl/shell_modules/detect/CT/dissimilarity.py) takes tasks (represented by a dictionary) as input, computes the Wasserstein Embedding of each task and outputs the pairwise Euclidean distance among the embeddings. Note that the image component of the input distribution is required to be normalized into `[-1,1]` and the action component has to be one-hot encoded, otherwise a preprocessing step will be applied to the input (by setting `one_hot=False` and `normalized=False`).

- [CT_graph](https://github.com/DMIU-ShELL/deeprl-shell/blob/main/deep_rl/shell_modules/detect/CT/CT_graph.ipynb) demonstrates how to calculate the dissimilarities among the randomly generated CT_graph tasks.
