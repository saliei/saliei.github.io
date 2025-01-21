---
title: "World Foundation Models"
date: "2025-01-19"
author: "Saeid"
description: "At CES 2025 event, NVIDIA announced, among other interesting things, including a single chip made out of\
    72 Blackwell GPUs, with 1.5 ExaFLOPS performance, it's first world foundation model, Cosmos."
---
At CES 2025 event, NVIDIA announced, among other interesting things, including a **single** chip made out of 
72 Blackwell GPUs, with **1.5 ExaFLOPS** performance, it's first world foundation model, [Cosmos](https://www.nvidia.com/en-gb/ai/cosmos/). 

DeepMind is working on a similar model, and this is also one of xAI's core missions. I suspect that these 
models are the next frontier in scientific and aritifical intelligence research, after AlphaFold's breakthroughs 
in protein folding prediction, and now Microsoft's MatterGen model, which is specifically tailored for matter science.

Reading from the [Cosmo's white paper](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai), 
it is designed to be *general purpose* simulator for physical AI systems, e.g. systems that deal with the phycial world, like robots, 
self-driving cars being one of them, it aims to be a *digital twin* of the physical world, albeit a limited one, at least at the moment.

The video curator handles about 20M of raw videos. Tokenizer has two modes, continuous tokenizer for diffusion models, 
and a descrete one for autoregressive models. As such it has two types of models, a diffusion-based model, where videos are generated 
by gradually denoising a random initial noise, which is better in video quality compared to the autoregressive model, where videos 
are created token by token, which could be faster at generation, both types are available in 4B and 12B parameters, which seems to be 
low amount, for a foundation model, and least of all, a *physical* foundation model, which I assume has much more free parameters compared 
to language.

It supports multimodel inputs (text, video, camera poses). It generates physically *plausible* videos, and can vastly improve the time to 
train physical AI systems, like robots and can be even fine-tuned on specific tasks. What genuinely interests me about these physical models, 
is this question:

**Are these models learning the physical laws?** The so called, **Physical Alignment** problem, at the moment with the current training methods 
based on customized transformer models, the simple answer is **NO**. But it's very interesting in all of the video generating models, for example 
simply by looking at a lot of, e.g. fluid flow videos, the model has learned that a fluid flows in a specific manner, that it has viscosity, etc.
it hasn't solved Navier-Stokes equations in one way or another, it simply has associated the word fluid to certain types of videos. 
it learns these patterns purely from observational data rather than incorporating actual physics principles or for example classic force calculations.

I imagine a child also doesn't solve Navier-Stokes equations for fluids to know that water will flow in a certrain way when spilled from a glass. 
I imagine what we need for a true physical foundation model, one that is aware of physical laws, or aims to construct a physical constructs, like physicist do.
This statement may not age well, and it turns out that the approach is completely different to the one we were doing so far, in training models with neural networks 
, at least the way we are doing that till now.

The authors of course acknowledge this: "Current models, including ours, fall short as reliable simulators of the physical world... our models still 
suffer from issues, including the lack of object permanence, inaccuracies in contact-rich dynamics...". They've tested this with free-falling objects, 
objects rolling down slopes, etc. One way to move forward maybe, like authors have suggested, to change the network architecture such that it could 
incorporate physical principles, or to actually use physics simulation engines.

