# Microbe2Pixel-SDM23

In recent years, machine learning (and particularly deep-learning) has gained significant attention in the biomedical domain. For example, deep-learning has become a go-to method for medical image analysis tasks. In other areas, such as metagenomics analysis, the application of deep-learning is still underdeveloped. This can be attributed to the tabular nature of metagenomics data, sparsity of features and the complexity, leading to (perceived) unexplainability, of deep-learning techniques.

Here, we propose Microbe2Pixel, a novel technique for implementing deep neural networks to fecal metagenomics data by incorporating taxonomic information in an image-based data representation. An important advantage of our method is the use of transfer learning, decreasing the number of samples needed for training. Furthermore, we develop a local model-agnostic feature importance algorithm that gives interpretable explanations which we evaluate in comparison to other local image explainer methods using quantitative (statistical performance) and qualitative (biological relevance) assessments.

Microbe2Pixel outperforms all other tested methods from both tested perspectives. The feature importance values have greater significance according to the current knowledge of microbiology and are more robust (regarding the number of samples used to train the model). This is especially significant for the use of deep-learning in smaller interventional clinical trials (e.g., fecal microbial transplant studies), where large sample sizes are not attainable.

Keywords: metagenomics, interpretable deep-learning, local explenations.
