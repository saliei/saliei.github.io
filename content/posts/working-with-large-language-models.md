---
title: "Working with Large Language Models"
date: "2025-02-15"
author: "Saeid"
description: "An overview of useful tools and workflows in working with LLMs."
draft: true
---

Large language models are experiencing their Cambrian explosion, and they may very well be the 
path to AGI, though with the current architectures and soley relying on scaling, may not be the solution.
In this article which I will try to keep updating it I inted to summarize useful workflows with them.

Like a lot of other breakthroughts, the authors of *Attention is all you need* perhapse initially didn't 
fully realize the full impact of their work. They introdoced **Transformers** model for machine translation, 
based on **self-attention**, **multi-head attention**, **positional encoding**, and **encode-decoder** concepts, 
which proved quite useful beyond translation taks, especially at scale because of inate parallel computation 
capabilities of the architecture.

- To download a GGUF model from HuggingFace hub:
```bash
# install huggingface-cli if not already
pip install huggingface-hub
# download from a repo
huggingface-cli download MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF --local-dir . --include '*Q2_K*gguf'
```

- Modelfile:
```
FROM ./downloads/mistrallite.Q4_K_M.gguf
```

```
ollama create mistrallite -f Modelfile
ollama run mistrallite "What is Grafana?"
```

- maybe agents with `llamaindex`:
[llamaindex](https://docs.llamaindex.ai/en/logan-material_docs/understanding/agent/basic_agent/)
