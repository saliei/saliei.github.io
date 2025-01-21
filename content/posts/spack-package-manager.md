---
title: "Spack Package Manager"
date: "2025-01-14"
author: "Saeid"
description: "Recently I gave a talk/mini workshop on spack package manager at SKAO, which forced me to dig into the core of spack.
	Spack boasts itself as the package manager for HPC, and it really is. The idea is straightforward, \
	streamline what we normally do when compiling and installing software manually on HPC systems using Python. \
	It's a simple idea but not at all easy to implement."
---
Recently I gave a talk/mini workshop on [spack](https://github.com/spack/spack) package manager 
at SKAO ([available here](https://saliei.io/spack-workshop)), for which I dug into the core of spack. 
Spack boasts itself as the package manager for HPC, and it really is. The idea is straightforward, 
streamline what we normally do when compiling and installing software manually on HPC systems using Python. 
It's a simple idea but not at all easy to implement. Compiling software is not an easy task, 
especially the scientific software and the ones that are designed to run on HPC systems on a large scale. 
Oftentimes, there are a lot of subtleties, and how it's compiled can significantly affect the performance. 
At the core of spack, there is the concretizer which builds the DAG (Directed Acyclic Graph) for the dependencies. 
The way it achieves this is interesting, package recipes are written in Python, and they subclass a specific build system, for example, CMake. 
The package dependencies with their constraints are written with `depends_on` directives, 
and they can be conditioned with a `when` argument, for example, 
a package may depend on the Intel Math Kernel Libraries (MKL) as its BLAS backend when specifically 
asked by the user with e.g. `+mkl` specifier in the command line, this will be then written as:

```python
depends_on("intel-onepi-mkl@2023.0.0:", when="+mkl")
```

Spack uses Clingo to do the concretization. Clingo is the most widely used ASP (Answer Set Programming) 
solver developed by the University of Potsdam that, uses SAT (Boolean Satisfiability) solver. 
ASP is a declarative programming that is used for searching and optimization problems. 
It's particularly good at handling knowledge representation, constraint satisfaction, 
and problems with multiple solutions (answer set). Clingo first uses gringo to ground the ASP program, 
which converts variables into propositional logic, then converts this into SAT clauses, 
and then uses the SAT techniques to determine if there's an assignment of boolean variables 
that makes the propositional logic formula true. Once the concretizer has determined 
the concrete specifications, it builds the DAG (Directed Acyclic Graph), 
where nodes represent packages and edges represent dependencies. 
Each node contains enough information about exactly what needs to be built, 
and the structure of the graph ensures the proper build order. 
The stages where the solver spends most of its time, 
and what is the optimization priority criteria, can be viewed:

```bash
# assume a fresh environment, don't reuse
$ spack solve --fresh --timers fftw
    setup          		3.160s
    load           		0.214s
    ground         		1.252s
    solve          		1.173s
    construct_specs     0.524s
    total          		6.488s

==> Best of 4 considered solutions.
==> Optimization Criteria:
  Priority  Criterion                                            Installed  ToBuild
  1         requirement weight                                           -        0
  2         number of packages to build (vs. reuse)                      -        0
  3         number of nodes from the same package                        -        0
  4         deprecated versions used                                     0        0
  5         version badness (roots)                                      0        0
  6         number of non-default variants (roots)                       0        0
  7         preferred providers for roots                                0        0
  8         default values of variants not being used (roots)            0        0
  9         number of non-default variants (non-roots)                   0        0
  10        preferred providers (non-roots)                              0        0
  11        compiler mismatches that are not required                    0        0
  12        compiler mismatches that are required                        0        0
  13        non-preferred OS's                                           0        0
  14        version badness (non roots)                                  0        1
  15        default values of variants not being used (non-roots)        0        1
  16        non-preferred compilers                                      0        0
  17        target mismatches                                            0        0
  18        non-preferred targets                                        0        0
  19        compiler mismatches (runtimes)                               0        0
  20        version badness (runtimes)                                   0        0
  21        non-preferred targets (runtimes)                             0        0
  22        edge wiring                                                  1        0

 -   fftw@3.3.10%apple-clang@16.0.0+mpi~openmp~pfft_patches+shared build_system=autotools patches=872cff9 precision=double,float arch=darwin-sequoia-m1
 - 	 	...
```

Interestingly, the generated ASP can also be viewed with the `--show=asp` flag (it's long, some 30000 or so facts!).
I think spack solves a hard problem, which has existed since the dawn of computers, in an interesting brute-force way.

PS. Some examples of spack packages are available in the [jupyterlab instance](https://saliei.io/jupyterlab).



