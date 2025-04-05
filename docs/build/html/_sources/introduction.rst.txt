Introduction
============

Welcome to **Neurostates** documentation!

**Neurostates** is a Python package that aims to standardize the analysis of consciousness states through recurrent brain connectivity patterns,
commonly referred to in the literature as "Brain States." The project provides tools for detecting recurrent brain states,
identifying their specific connectivity patterns, and estimating their probability of occurrence based on preprocessed neuroimaging data,
specifically Electroencephalography (EEG) and functional Magnetic Resonance Imaging (fMRI).

What You Can Do with **Neurostates**
------------------------------------

- Compute dynamic functional connectivity using different methods (e.g., Pearson, Cosine similarity, Spearman R, or a custom connectivity).
- Cluster functional connectivity matrices (e.g., k-means or a custom clustering algorithm).
- Rank brain states based on entropy.
- Construct the probability distribution of brain states for each subject.

Why Use **Neurostates**?
------------------------

**Neurostates** is designed to be both **easy to use** and **highly flexible**. It leverages scikit-learn estimators and pipelines,
allowing seamless processing from one step to the next while enabling access to intermediate results.
Users can get started quickly with default settings, but the framework also supports customization, allowing the use of custom
connectivity metrics, clustering algorithms, and other analysis methods to suit specific research needs.

Getting Started
---------------

To get started with **Neurostates**, follow the :doc:`installation guide <installation>`, and you can start using the modules straight away.
The documentation will guide you through the setup and provide examples on how to use the most common features.

If you have any questions, feel free to check out our Github issues or write us an email to: dellabellagabriel@gmail.com or natirodriguez114@gmail.com
