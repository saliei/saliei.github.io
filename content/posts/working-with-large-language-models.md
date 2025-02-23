---
title: "Working with Large Language Models"
date: "2025-02-15"
author: "Saeid"
description: "Large language models are experiencing their Cambrian explosion. 
    They may not be the path to AGI, but at least they give a taste of what it could be. 
    The current mainstream approach to rely on scaling, may not be the sole solution, 
    as the data runs out, and the models are plateauing in benchmarks, 
    though we may very well see emergent behaviors that surprise us, 
    as the bitter lesson taught us. In this article, I first review Transformers model, 
    then summarize useful workflows with LLMs, which I intend to keep updating."
---
Large language models are experiencing their Cambrian explosion. 
They may not be the path to AGI, but at least they give a taste of what it could be. 
The current mainstream approach to rely on scaling, may not be the sole solution, 
as the data runs out, and the models are plateauing in benchmarks, 
though we may very well see emergent behaviors that surprise us, 
as the [bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) taught us. 
In this article, I first review Transformers model, then summarize useful workflows with LLMs, 
which I intend to keep updating.

Like many other breakthroughs, the authors of *Attention is all you need*, 
perhaps initially didn't fully realize the full impact of their work. 
They introduced the **Transformers** model for machine translation, based on **self-attention**, **multi-head attention**, 
**positional encoding**, and **encoder-decoder** concepts, which proved quite useful beyond translation tasks, 
especially at scale, because of the innate parallel computation capabilities of the architecture. 
This post is not meant to be an introduction to transformers, though here we review the core concepts.

## Transformers

A transformer consists of an encoder-decoder structure, though many models (e.g., BERT, GPT) use only one part.
- The encoder maps an input sequence \$x = (x_1, x_2, …, x_n)\$, to a sequence of hidden representations.
- The decoder takes these representations and generates an output sequence \$y = (y_1, y_2, …, y_m)\$, step by step.

Each encoder and decoder consists of multiple stacked layers, each with a self-attention mechanism. 
The decoder also includes **cross-attention**, allowing it to attend to encoder outputs.

### Self-Attention Mechanism

Self-attention is at the core of transformers, allowing each token in a sequence to dynamically focus on relevant tokens. 
The key idea is to compute attention scores that define how much focus each word should give to every other word.

For an input sequence represented as an embedding matrix \$X \in \mathbb{R}^{n \times d}\$  
(where  \$n\$  is the sequence length and  \$d\$  is the embedding dimension), self-attention works as follows:

1. Compute three learned projections, **Query**, **Key**, and **Value** matrices for the input tokens:

$$
\begin{align}
Q &= XW_Q, \\ 
K &= XW_K, \\ 
V &= XW_V
\end{align}
$$

Where:
- \$Q\$ (Query): Represents what this token is "searching for" in others.
- \$K\$ (Key): Represents what this token "has to offer."
- \$V\$ (Value): Represents the actual content of the token.
- \$W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}\$ are learnable weights, projection matrices, 
where \$d_k\$ is the dimension of queries and keys.

2. Compute attention scores between all tokens using a dot product:

$$ 
\text{Scores} = QK^{T} 
$$

this results in a matrix \$\mathbb{R}^{n \times n}\$, where each entry, \$S_{ij}\$ 
represents the relevance of token \$i\$ to token \$j\$.

3. Scale the cores to prevent large values that can lead to vanishing gradients, 
we scale the scores by \$\sqrt{d_k}\$:

$$ 
\text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}} 
$$

Where \$d_k\$ is the dimension of queries and keys. 
The scaling prevents extreme values that could saturate the softmax function.

4. Apply softmax to get attention weights. A softmax function is applied row-wise 
to normalize the scores into probabilities:

$$ 
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) 
$$

Each row sums to 1, so every token's attention distribution is a well-defined probability.

5. Compute the weighted sum of values. Each token's final representation is a weighted sum of the values:

$$
\text{Output} = AV
$$

Where \$A\$ (attention matrix) determines how much focus each token places on others.


Self-attention enables transformers to model relationships between tokens in a sequence. 
It allows each token in a sequence to attend to, or focus on, other tokens, dynamically weighing their importance. 
Unlike RNNs, which process tokens sequentially, self-attention enables parallel computation across all tokens at once. 
Given an input sequence of $n$ tokens, each token computes its attention score with every other token, 
determining how much "attention" it should give to them.
 
#### Multi-Head Attention

Instead of a single attention function, transformers use **multi-head attention**, 
where \$h\$ separate attention heads are computed in parallel and concatenated. 
Each head computes:

$$
\text{head}_i = \text{Attention}(XWQ_i, XW_{K_i}, XW_{V_i})
$$

The outputs from all heads are concatenated and linearly transformed: 

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, …, \text{head}_h) W_O
$$

Where \$W_O\$ is a final learnable projection matrix. 
Multi-head attention enables learning of different attention patterns, 
allowing the model to capture diverse relationships between tokens.

### Positional Encoding

Transformers do not have recurrence or convolution, so they need explicit positional encodings 
to incorporate token order information. One alternative would be to learn a separate positional 
embedding vector for each position, just like word embeddings. 
However, this approach does not generalize well to longer sequences than those seen during training. 
Instead, the original authors of the paper proposed a **fixed sinusoidal encoding**. 
For a token at position \$p\$, its encoding is:

$$
\begin{align}
\text{PE}_{(p, 2i)} &= \sin\left(\frac{p}{10000^{2i/d}}\right) \\
\text{PE}_{(p, 2i+1)} &= \cos\left(\frac{p}{10000^{2i/d}}\right)
\end{align}
$$

Where:
- \$p\$ is the position index (e.g. first token = 0, second token = 1, etc.).
- \$i\$ indexes the dimension of the embedding.
- \$d\$ is the total embedding dimension.

The sinusoids enable:
- Smoothness: Nearby positions have encodings that are close to each other.
- Long-Range Generalization: Since sine and cosine functions are periodic, they help generalize to longer sequences than those during training.
- Unique Encoding: Each position has a unique encoding.

After computing the positional encodings, they are added to the word embeddings:

$$
X^\prime = X + \text{PE}
$$

with this, the transformer sees both content (word embedding) and position (encoding).

### Feed-forward network and Normalization

Each transformer layer contains a position-wise feed-forward network (FFN) applied independently to each token:


$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$


where \$W_1\$ and \$W_2\$ are learnable weight matrices.

Additionally, *LayerNorm* and *residual connections* are used to stabilize training:

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

### Variants

Several transformer models modify the basic architecture:
- BERT (Bidirectional Encoder Representations from Transformers):
    - Uses only the encoder part.
    - Trained with masked language modeling (MLM) and next sentence prediction (NSP).
    - Used for classification, question answering, and sentence embedding.
- GPT (Generative Pre-trained Transformer):
    - Uses only the decoder part.
    - Trained with causal (autoregressive) language modeling.
    - Used for text generation.
- T5 (Text-to-Text Transfer Transformer):
    - Uses a full encoder-decoder.
    - Reformulates all tasks as text-to-text.
- Vision Transformers (ViT):
    - Applies transformers to images by splitting them into patches.
    - Uses self-attention for image classification.

## Local LLMs

There have been many open-source projects trying to streamline the process. 
One of the most useful projects, is the [llama.cpp](https://github.com/ggml-org/llama.cpp) 
initially developed by Georgi Gerganov, which has now spurred a lot other utilities built around it.

To run the the local models, after downloading the model parameters in `GGUF` format, 
which is a binary format to store inference models optimized for consumer-grade computers 
([read here](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) more about it), 
install `llama.cpp` which comes with a rich set of CLI utilities, like `llama-server`:

```bash
# install huggingface-cli to download models easily from hugginface
pip install huggingface-cli
# download from a repo with the desired quantization variant
huggingface-cli download MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF --local-dir models/ --include '*Q2_K*gguf'
# install llama.cpp and start llama-server
llama-server -m models/Meta-Llama-3-8B-Instruct.Q2_K.gguf --port 8080
```

There is also the python bindings, [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), but we can 
query the llama server ourselves:

```python

server_url = "http://127.0.0.1:8080"
system_prompt = "you are a helpful assistant"

async def query(message):
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            *message
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }

    async with aiohttp.ClientSession() as session:
        try:
            url = f"{server_url}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    return None
        except Exception as e:
            logger.exception(f"exception occrred while querying server: {e}")
            return None

prompt = "What is GGUF format specification?"
message = [{"role": "user", "content": prompt}]

response = await query(message)
print(response)
```

There is also [ollama](https://ollama.com/), which has a models hub:

```bash
ollama run deepseek-r1:7b
# or just pull model GGUF under ~/.ollama
ollama pull deepseek-r1:7b
```

Ollam also supports configuring and creating models with a 
[`Modelfile`](https://github.com/ollama/ollama/blob/main/docs/modelfile.md) 
like a `Dockerfile`:

```
FROM llama3.2

PARAMETER temperature 1
PARAMETER num_ctx 4096

SYSTEM You are a helpful assistant.
```

Then just create a model and query it:

```bash
ollama create model-name -f Modelfile
ollama run model-name "What is GGUF?"
```

## Collection

Here is a collection of utilities which I find useful:

- [r/LocalLLaMa](https://www.reddit.com/r/LocalLLaMA/): Local LLMs subreddit.
