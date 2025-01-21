---
title: "World Foundation Models"
date: "2025-01-19"
author: "Saeid"
description: "At CES 2025 event, NVIDIA announced, among other interesting things, including a single chip made out of\
    72 Blackwell GPUs, with 1.5 ExaFLOPS performance, it's first world foundation model, Cosmos."
---
At CES 2025, NVIDIA announced several interesting projects, 
including a single chip of 72 Blackwell GPUs, with 1.5 ExaFLOPS FP4 performance!
Among the announcements was its first world foundation model, 
[Cosmos](https://research.nvidia.com/publication/2025-01_cosmos-world-foundation-model-platform-physical-ai). 

DeepMind is working on a similar model, and this is also one of xAI's core missions. 
I suspect that these models are among the next frontier in scientific and AI research, 
following AlphaFold's breakthroughs in protein folding prediction, 
and now Microsoft's MatterGen model, which is specifically tailored for material science.

Reading from Cosmo's white paper, 
it is designed to be a *general-purpose* simulator for physical AI systems, 
e.g. systems that deal with the physical world, like robots, 
and I take self-driving cars to be also robots. 
It aims to be the *digital twin* of the physical world, though understandably limited currently.

The model tokenizer has two modes: a continuous tokenizer for diffusion models, 
and a discrete one for autoregressive models. The diffusion-based model generates videos 
by gradual denoising of a random initial noise, which is better in video quality 
compared to the autoregressive model, which generates videos by predicting the next token, 
which could be faster than diffusers. Both model types are available in 4B and 12B parameters, 
which seems limited for a foundation model, least of all for a *physical* foundation model, 
which I'd expect has much more free parameters compared to language models.

The model generates physically *plausible* videos that improve the training time 
of physical AI systems by orders of magnitude and can even be fine-tuned on specific tasks. 
What genuinely interests me about these models is the following question:

**Are these models truly learning the laws of physics?** \
The so-called, **Physical Alignment** problem, 
and if not, **how can we do that?** 

Currently with the training methods based on transformer models, 
the simple answer is **NO**, but it's interesting! The video-generating models, 
simply by looking at a lot of, e.g. fluid flow videos, have learned patterns 
that fluid will flow in a specific manner, that it has a property we call viscosity, etc. 
It hasn't solved Navier-Stokes equations in one way or another, 
it simply has associated the word fluid to certain types of videos. 
It learns these patterns purely from observational data rather than incorporating actual physics principles 
or for example classical force calculations.

I imagine a child doesn't also solve Navier-Stokes equations to have an intuition that 
water will flow in a certain way when spilled from a glass. To achieve a true physical foundation model, 
I suspect that this model one way or another has to have an understanding of the standard model, 
we might need a fundamentally different approach, or architectures that have true physics-informed learning, 
e.g. by incorporating physics engines in the training.

The authors of the Cosmo's paper acknowledge this limitation:\
"*Current models, including ours, fall short as reliable simulators of the physical world... 
our models still suffer from issues, including the lack of object permanence, 
inaccuracies in contact-rich dynamics...*".\
They've tested the model with scenarios like objects free-falling and rolling down slopes, highlighting it's shortcomings.

One way to move forward, as authors have suggested, is maybe to change the network architecture such that it could 
incorporate physical principles at its core, or use a physics simulation engine. 
One thing is certain, the future of physical foundation models is much brighter than what it is now!

