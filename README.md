# UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities
  
[![Paper](https://img.shields.io/badge/arXiv-2504.20734-b31b1b)](https://arxiv.org/abs/2504.20734)
[![Project-Page](https://img.shields.io/badge/Project-Page-green)](https://universalrag.github.io)

> 🚀 We will release the full codebase soon — stay tuned!

<img src="assets/concept.png" alt="Concept Figure" width="80%">

## Abstract

Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing RAG approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, a novel RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single combined corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose a modality-aware routing mechanism that dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it. Also, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 8 benchmarks spanning multiple modalities, showing its superiority over modality-specific and unified baselines.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{yeo2025universalrag,
  title={UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities},
  author={Yeo, Woongyeong and Kim, Kangsan and Jeong, Soyeong and Baek, Jinheon and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2504.20734},
  year={2025}
}
```