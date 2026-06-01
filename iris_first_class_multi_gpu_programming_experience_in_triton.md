                                             Iris: First-Class Multi-GPU Programming Experience in Triton
                                                     Muhammad Awad                                 Muhammad Osama                                Brandon Potter
                                                    muhaawad@amd.com                           muhammad.osama@amd.com                       brandon.potter@amd.com
                                                 Advanced Micro Devices, Inc.                  Advanced Micro Devices, Inc.                Advanced Micro Devices, Inc.
                                                    Santa Clara, CA, USA                          Santa Clara, CA, USA                          Austin, TX, USA
                                         Abstract                                                                architecture, both of which vary substantially in real-world deploy-
                                         Multi-GPU programming traditionally requires developers to navi-        ments. As a result, collective communication libraries must deliver
                                         gate complex trade-offs between performance and programmabil-           high performance across diverse and evolving environments.
                                         ity. High-performance implementations typically rely on low-level          Given this complex landscape, higher-level abstractions are es-
                                         HIP/CUDA communication libraries that demand substantial en-            sential to simplify development without sacrificing performance.
arXiv:2511.12500v1 [cs.DC] 16 Nov 2025




                                         gineering effort for even basic overlap patterns, while simpler ab-     However, modern accelerators also introduce specialized units and
                                         stractions often sacrifice performance. We present Iris, a multi-GPU    mechanisms—such as tensor cores and asynchronous memory-
                                         communication library implemented entirely in Python and Triton         movement engines (SDMA, TMA, and asynchronous copy instruc-
                                         that eliminates this trade-off. Iris provides tile-based symmetric      tions)—that demand fine-grained, tile-based programming to fully
                                         memory abstractions that naturally align with Triton’s program-         exploit their capabilities for overlapping communication and com-
                                         ming model, enabling developers to write single-source kernels          putation. A number of efforts have advanced the state of the art,
                                         that seamlessly interleave computation and communication. We            including compiler-driven approaches (e.g., XLA [18], TVM [7],
                                         demonstrate a taxonomy of compute-communication overlap pat-            and Triton [26]) as well as template-based libraries (e.g., CUTLASS,
                                         terns—from bulk-synchronous to fine-grained workgroup special-          CuTe [10], ThunderKittens [24]). Among these, Triton has shown
                                         ization—that can be implemented with minimal code changes in            sustained maturity, performance portability, and broad adoption
                                         Iris, often requiring just a few additional lines within the same       for computation. However, a critical gap remains: while these ab-
                                         Triton kernel. Our evaluation shows that Iris achieves near-optimal     stractions have successfully tackled local operators and compute
                                         bandwidth utilization in microbenchmarks and delivers up to 1.79×       kernels, they have largely overlooked communication as an equally
                                         speedup over PyTorch and RCCL for GEMM+All-Scatter workloads,           important concern. As distributed training and inference scales
                                         demonstrating that high-level implementations can match or ex-          to hundreds or thousands of GPUs, communication becomes the
                                         ceed heavily-optimized libraries while dramatically simplifying         dominant performance bottleneck.
                                         multi-GPU programming.
                                                                                                                    The Consensus: Fine-Grained Overlap is Essential. The research
                                         Keywords                                                                and production communities have converged on a clear solution:
                                         Distributed Computing, GPU, Fused Kernels, Triton, Wavefront-           overlap communication with computation at fine granularity. Rather
                                         specialization                                                          than executing communication and computation in rigid, sequential
                                                                                                                 bulk-synchronous phases (Figure 1a), modern workloads demand
                                         1   Introduction                                                        fine-grained overlap where data is communicated as soon as it
                                         Modern AI workloads demand near-peak performance to extract             is produced, at tile granularity (Figure 1b), so that computation
                                         the full efficiency of AI systems. Teams of specialists with deep       can proceed on other tiles without waiting. This fine-grained over-
                                         understanding of both model characteristics and hardware architec-      lap hides communication latency behind useful work, eliminating
                                         ture are required to craft highly optimized training and inference      the idle “bubbles” that plague bulk-synchronous execution. Recent
                                         kernels for these AI workloads. Even for seasoned engineers, this       systems such as TorchTitan’s AsyncTP [11] and production LLM
                                         process requires iterative refinement, hardware-specific tuning, and    training pipelines [8, 23, 27] have demonstrated the necessity of
                                         extensive experimentation. This challenge arises because contem-        this approach.
                                         porary AI models comprise numerous operators, each with multiple
                                         potential optimization strategies. Determining the appropriate op-         Communication: The Missing Abstraction. Fine-grained overlap
                                         timizations depends on a wide range of factors—including model          requires a communication abstraction that operates at the same tile
                                         structure, configuration parameters, hardware vendor and genera-        granularity as Triton’s computational model. However, Triton pro-
                                         tion, system topology, and supporting software ecosystems. More-        vides no such abstraction. While Triton’s tile-based programming
                                         over, end-to-end performance is strongly influenced by distributed-     model has revolutionized how developers write optimized compute
                                         parallel execution primitives—such as all-reduce and all-to-all—that    kernels—automatically handling memory coalescing, shared mem-
                                         orchestrate computation across devices.                                 ory management, and intra-kernel scheduling–communication re-
                                             Distributed parallelism further amplifies the complexity. Prac-     mains an afterthought. Developers must rely on external libraries
                                         titioners routinely combine data, tensor, pipeline, and expert par-     (RCCL [4], NCCL [13]) that operate at coarse kernel boundaries,
                                         allelism strategies and expect these hybrid approaches to perform       or hand-craft device-to-device data movement without compiler
                                         consistently across heterogeneous hardware platforms. For system        support. This approach forces communication to remain outside
                                         and library developers, this presents a significant challenge: kernel   the compiler’s purview, preventing the tile-level interleaving that
                                         efficiency is tightly coupled to network characteristics and system     the consensus approach demands. The result is that fine-grained
                                                                                                                                                      Awad, Osama and Potter


             CPU          GPU1       GPU2        GPU3           CPU          GPU1       GPU2   GPU3
                                                                                                      workloads (Figure 1b). Unlike existing approaches that wrap legacy
                                                                                                      SHMEM libraries as opaque bytecode, Iris is implemented entirely
                                                                                                      in Python and Triton, giving the compiler full visibility into both
                                                                                                      computation and communication. Iris provides native symmetric
                                                                                                      memory abstractions that enable developers to write concise single-
                                                                                                      source kernels that seamlessly interleave computation and com-
      TIME




                                                        TIME
                                                                                                      munication at tile granularity with no external dependencies, no
                                                                                                      opaque library calls, no bulk-synchronous phases. Iris also offers
                                                                                                      an experimental Gluon backend using Triton’s @gluon.jit and
                                                                                                      @aggregate decorators for improved ergonomics, but this paper
                                                                                                      focuses on the standard Triton API for clarity and broader compati-
                                                                                                      bility. Our contributions are as follows:
              CPU Execution                                      CPU Execution                              • Native Triton implementation: The first multi-GPU com-
              GPU Kernel Execution                               GPU Kernel Execution
              Communication Kernel Execution                     Stream Synchronize                            munication library implemented entirely in Python and
              Stream Synchronize
              GPU-CPU Channel Operations
                                                                                                               Triton, providing full compiler visibility and enabling co-
                                                                                                               optimization of computation and communication
             (a) Bulk-Synchronous                              (b) Fine-Grained Overlap                     • Tile-based symmetric memory API: Pythonic abstrac-
4 |                                        5 |

                                                                                                               tions that align naturally with Triton’s tile-centric pro-
      Figure 1: Execution model comparison: (a) bulk-synchronous                                               gramming model, supporting both value-based and pointer-
      execution with rigid sequential phases and CPU-initiated                                                 based communication primitives
      kernel launches versus (b) fine-grained overlap enabling dy-                                          • Taxonomy of fused patterns: A comprehensive classifica-
      namic communication at tile granularity with GPU-initiated                                               tion of compute-communication overlap strategies, includ-
      execution.                                                                                               ing bulk-synchronous, producer-consumer, and workgroup-
                                                                                                               specialized approaches
      overlap, despite being widely recognized as essential, remains inac-                                  • Performance validation: Experimental evaluation demon-
      cessible to Triton developers.                                                                           strating up to 1.79× speedup over PyTorch and RCCL for
                                                                                                               GEMM+All-Scatter workloads across multiple problem sizes
         The Challenge: Architectural Limitations of Existing Approaches.                                   • Open-source release: Fully open-source implementation
      Recognizing this gap, several efforts have attempted to bring com-                                       enabling reproducibility and community adoption
      munication into Triton. However, these attempts face fundamental
      architectural constraints. Current efforts typically wrap vendor                                2     Background and Related Work
      libraries (rocSHMEM [3], NVSHMEM [16]) as opaque bytecode,                                      Iris builds on established GPU communication mechanisms—symmetric
      linking them into Triton kernels. While pragmatic, this approach                                memory, direct inter-GPU interconnects, and well-defined mem-
      introduces several limitations. First, these libraries inherit design                           ory consistency models—while introducing a novel programming
      constraints from the OpenSHMEM specification and carry techni-                                  abstraction that distinguishes it from prior work. We first review
      cal debt from APIs originally designed for CPU-based distributed                                the underlying hardware and runtime infrastructure, then survey
      computing, which do not align naturally with Triton’s tile-centric                              related efforts to integrate communication into GPU programming
      programming model and introduce anti-patterns that work against                                 frameworks.
      compiler optimizations. Second, and more critically, wrapping exter-
      nal libraries as opaque binaries prevents the compiler from seeing                              2.1    Background
      communication operations, precluding co-optimization of com-
                                                                                                      We briefly review the key mechanisms that enable multi-GPU com-
      putation and communication, intelligent scheduling, and unified
                                                                                                      munication on AMD platforms. This includes the physical intercon-
      fusing of kernels across their boundaries. Communication remains
                                                                                                      nect topology, the runtime interfaces for establishing symmetric
      a second-class citizen—linked as binary blobs rather than first-
                                                                                                      memory across processes, and the memory consistency model that
      class Triton code. Similarly, traditional collective communication
                                                                                                      provides correctness guarantees. Together, these components form
      libraries (CCLs) such as RCCL and NCCL impose bulk-synchronous
                                                                                                      the execution substrate on which Iris is built.
      semantics with CPU-initiated kernel launches, adding coordination
      overhead, kernel teardown costs, and redundant memory transfers                                 2.1.1 Interconnect Topology. Iris targets scale-up environments
      at kernel boundaries. These architectural constraints limit the abil-                           where multiple GPUs within a single node are connected via a high-
      ity to achieve the native, compiler-visible, tile-granular primitives                           bandwidth interconnect. The AMD Instinct MI300X and MI325X
      required for true fine-grained overlap.                                                         platforms [2] use seven high-bandwidth, low-latency AMD Infinity
                                                                                                      Fabric links per GPU to form a fully connected 8-GPU system.
         Iris: Native Communication for Tile-Based Programming. We present
                                                                                                      Each GPU is also connected to the host CPU via a x16 PCIe Gen
      Iris [5]1 , the first multi-GPU library architected from the ground
                                                                                                      5 link. This fully connected mesh topology provides direct peer-
      up for Triton’s tile-based programming model to full enable the
                                                                                                      to-peer access between any GPU pair without traversing the host
      fine-grained computation and communication overlap within AI
                                                                                                      CPU, enabling the low-latency, high-bandwidth communication
      1 Iris is open-source and available at https://github.com/ROCm/iris                             that Iris leverages for efficient multi-GPU operations. Iris exploits
Iris: First-Class Multi-GPU Programming Experience in Triton


this topology to implement efficient collective operations and point-   Table 1: Comparison of multi-GPU communication ap-
to-point communication patterns directly in Triton kernels.             proaches.

2.1.2 Symmetric Memory via IPC. Iris establishes symmetric mem-          Aspect                Built for Triton            Wrapper-Based
ory across GPUs using HIP’s inter-process communication (IPC)
                                                                         Libraries             Iris                        Triton-Dist., PyTorch
mechanism [1]. Each process allocates device memory using stan-
dard hipMalloc, then exports handles via hipIpcGetMemHandle              Approach              Native Triton               Wraps xSHMEM

and imports peer handles via hipIpcOpenMemHandle. This enables           Programming Model     Triton-native               OpenSHMEM-based
direct memory access across GPU boundaries: each GPU can read            Compiler Visibility   Full, co-optimization       Opaque, limited
from and write to any peer GPU’s memory using simple pointer             API Style             Pythonic, tile-based        C-style, Python-wrapped
arithmetic. Iris uses coarse-grained memory semantics for sim-           Language              Python + Triton             Python + xSHMEM
plicity and portability, relying on the memory consistency model
                                                                         Memory Model          C++/HIP model               Ill-defined
(described below) to ensure correctness of cross-GPU operations.
By leveraging IPC, Iris exposes a clean symmetric memory abstrac-        Synchronization       Acquire/release semantics   quiet/wait_until

tion to programmers, enabling Triton kernels to perform remote
memory operations as naturally as local ones.
                                                                        Iris reimagines the programming interface with a modern, Pythonic
2.1.3 Memory Model and Synchronization. Iris relies on AMD’s            API built natively in Triton. This approach preserves the conceptual
memory model to provide formal correctness guarantees in multi-         benefits of symmetric memory while eliminating the legacy con-
GPU execution. This model has been described in publicly avail-         straints of C-style interfaces and explicit thread-ID management
able literature [9] and is implemented concretely in AMDGPU             that were inherited from CPU-era HPC programming models.
LLVM [12]. The AMD memory model is Sequentially Consistent
Heterogeneous Race Free (SC-HRF), analogous to the C++ model but        2.2     Related Work
extended with GPU-specific memory scopes. The model supports            A number of systems provide abstractions for multi-GPU commu-
standard C++ memory orderings (acquire, release, acq_rel, and           nication, symmetric memory, and fused compute-communication
seq_cst) with familiar semantics: acquire operations prevent sub-       execution models. Table 1 contrasts Iris with existing approaches,
sequent loads and stores from being reordered before them, while        highlighting two fundamental implementation strategies: native
release operations prevent preceding loads and stores from being        implementations architected for the target programming model
reordered after them. Critically, the model introduces hierarchi-       versus wrapper-based approaches that integrate external libraries.
cal memory scopes—wavefront (warp), workgroup (block), agent            Below, we discuss the most closely related efforts in detail.
(device), and system—that define visibility domains for synchro-
                                                                        2.2.1   HIP/CUDA/C++-Based Approaches.
nization operations. Triton already exposes these memory orderings
and scopes through its atomic operations API, and Iris leverages           xSHMEM Libraries. As discussed in Section 2.1.4, xSHMEM li-
Triton’s implementation to use hardware-level synchronization           braries (NVSHMEM, rocSHMEM) implement the OpenSHMEM
primitives directly. For multi-GPU communication, Iris uses agent-      specification [17] and establish the symmetric memory abstraction
scoped atomics to synchronize between GPUs within a node, and           that Iris builds upon. However, their APIs do not align well with
system-scoped operations when broader visibility is required.           Triton’s tile-centric programming paradigm: they require explicit
   Iris adopts this memory model because it is well-established,        thread-ID management and rely on low-level C-style interfaces.
widely-adopted across CPU and GPU programming, and provides             These abstractions were originally built for HPC-CPU environ-
programmers with familiar, intuitive synchronization primitives         ments and later ported to GPU, which limits their effectiveness for
(acquire/release, memory scopes) that are straightforward to reason     modern GPU programming models. Iris retains the symmetric heap
about and use, while still delivering provable correctness guarantees   abstraction but provides a modern, Pythonic API that integrates
for multi-GPU coordination.                                             naturally with Triton’s tile-based execution model.
2.1.4 Symmetric Memory Programming Models. The symmetric                   Flux. Chang et al. [6] introduced Flux, which targets symmet-
memory programming model, established by the OpenSHMEM                  ric memory communication but is implemented directly in CUDA
specification and implemented by hardware vendors as xSHMEM             and CUTLASS. While offering highly optimized kernels, this ap-
variants (rocSHMEM [3] and NVSHMEM [16]), provides a foun-              proach requires longer development cycles and relies heavily on
dational abstraction where each process allocates memory in a           C++ template metaprogramming. Iris maintains a simpler Python-
symmetric heap that is directly accessible by all peers. This model     and Triton-based design that enables faster prototyping, easier
enables one-sided communication primitives—such as remote puts,         debugging, and full compiler visibility.
gets, and atomic operations—that can be issued without requiring
                                                                        2.2.2   Triton-Based Wrappers.
explicit receiver-side coordination, making it well-suited for GPU
programming where fine-grained producer-consumer patterns are              Triton-Distributed. Zheng et al. [28] introduced Triton-Distributed,
common.                                                                 which wraps existing xSHMEM libraries behind Triton-specific
   Iris adopts this symmetric heap abstraction as its core memory       Python APIs, introducing proof-of-concept fused kernels for GEMM
model, recognizing its proven utility for multi-GPU communication.      and AllGather and MoE-style patterns. While similar in motiva-
However, rather than wrapping existing xSHMEM implementations,          tion to Iris, it inherits the limitations of its underlying SHMEM
                                                                                                                        Awad, Osama and Potter


implementation: the communication layer remains a thin wrapper              Both HIP/HSA [9] and CUDA [15] programming models define
around vendor libraries, preventing compiler-level visibility. In con-   clear semantics for atomic operations with configurable memory
trast, Iris implements remote memory operations directly in Triton       ordering (relaxed, acquire, release, acquire-release) and synchro-
with no external dependencies, enabling full compiler visibility for     nization scopes (block, GPU, system). These well-defined scopes
fine-grained fused operations.                                           allow developers to precisely control the visibility and ordering
                                                                         of memory operations across different granularities—from thread
   PyTorch Symmetric Memory and TorchTitan AsyncTP. Recent Py-
                                                                         block synchronization to system-wide coherence across GPUs. This
Torch efforts [20] introduce symmetric memory support at the
                                                                         familiarity extends to both host-side APIs (PyTorch-compatible
framework level, enabling asynchronous tensor-parallel communi-
                                                                         tensor operations) and device-side APIs (Triton-native operations),
cation via decomposed point-to-point operations (put/get). TorchTi-
                                                                         which we detail in subsequent sections.
tan leverages this for fused GEMM-AllGather and MoE patterns op-
timized via torch.compile. However, similar to Triton-Distributed,
these systems rely on vendor-supplied communication backends,            3.1.3 Pure Python and Triton Implementation. A key distinguish-
limiting compiler optimization opportunities. Iris draws inspiration     ing feature of Iris is its implementation: the entire framework is
from the decomposed pattern approach but exposes primitives di-          built from scratch in Python and Triton without requiring external
rectly within Triton kernels, enabling tile-level fused kernels and      communication libraries or custom runtime dependencies. Unlike
eliminating kernel-switching overhead while avoiding reliance on         wrapper-based approaches such as Triton-Distributed (discussed
external communication libraries.                                        in Section 2.2) that rely on rocSHMEM bytecode or other low-level
                                                                         communication primitives, Iris leverages only the existing PyTorch
3     Iris                                                               ecosystem (for host-side operations) and HIP runtime APIs (for
Iris is a multi-GPU library built from scratch for scaling with min-     GPU IPC, as described in Section 2.1.2). This design choice provides
imal dependencies (only Triton, PyTorch, and HIP runtime). The           several advantages: (1) portability across different GPU vendors
library provides intuitive and simple APIs for developers without        without vendor-specific communication libraries, (2) ease of de-
requiring knowledge of distributed systems architecture, enabling        bugging and modification since the entire codebase is in high-level
Python and Triton developers to write multi-GPU code leveraging          Python and Triton, (3) simplified deployment with no additional
high-level language abstractions. First we will discuss the design       system dependencies beyond PyTorch and the standard HIP/CUDA
decisions that influenced Iris’s design and successfully resulted in     runtime, and (4) compiler visibility—the Triton compiler has full
an abstraction that allows Triton developers to be productive and        visibility into the entire codebase, enabling optimizations across
use familiar abstractions.                                               computation and communication boundaries rather than treating
                                                                         communication primitives as opaque binary blobs linked into the
3.1    Design Philosophy                                                 final executable.
Iris adopts the symmetric heap abstraction from SHMEM (as dis-
cussed in Section 2.1.4) but modernizes the programming model            3.1.4 Value- and Pointer-based APIs. Existing SHMEM-based APIs
for GPU computing. Rather than porting legacy SHMEM APIs, we             treat the source and destination arguments to a SHMEM func-
provide Pythonic and Triton-native interfaces that respect modern        tion as buffers that are pointed to by a pointer, along with their
programming paradigms. As summarized in Table 1, this distin-            respective buffer sizes. CPU threads are more heavyweight and typ-
guishes Iris from wrapper-based approaches that rely on existing         ically work on buffers of data, which likely influenced this design
xSHMEM implementations.                                                  choice. However, Iris targets GPUs where hundreds of thousands of
                                                                         threads are actively doing work. Massively-parallel GPUs require
3.1.1 Adoption of Symmetric Heap Abstraction. Iris implements
                                                                         both value-based and pointer-based operations. Value-based data
a Symmetric Heap abstraction. While Iris deviates from SHMEM-
                                                                         movement copies data directly from registers to other GPUs. In
like library APIs, it implements one of the core ideas adopted in
                                                                         contrast, pointer-based data movement acts as a data copy between
the OpenSHMEM specification: the symmetric heap. Symmetric
                                                                         the main memory of local GPUs and that of another GPU.
heaps are simple to understand, implement, and use. The symmetric
                                                                            The rise of tile-based programming frameworks such as Triton,
heap design provides predictable memory layouts across all GPUs,
                                                                         ThunderKittens [25], and CuTe DSL [14] demonstrates the impor-
enabling efficient pointer translation with minimal overhead since
                                                                         tance of value-based APIs that directly operate on tensors rather
each rank’s memory has an identical structure at corresponding
                                                                         than raw bytes. These frameworks prioritize high-level tensor ab-
offsets.
                                                                         stractions because they align with how developers reason about
3.1.2 Familiar Memory Model and Pythonic APIs. Rather than in-           computation and data movement in modern GPU programming.
troducing new synchronization primitives or memory semantics,            Iris’s value-based APIs enable developers to express operations
Iris adopts the well-established C++/HIP/CUDA memory model               at the granularity of computational tiles, moving partial results
with acquire/release ordering (as detailed in Section 2.1.3). This de-   directly from registers to remote memory without intermediate
sign choice is intentional: GPU programmers already reason about         buffering. This approach is particularly effective for fine-grained
memory consistency, ordering, and synchronization in their single-       computation-communication overlap patterns where data becomes
GPU kernels. By reusing these familiar semantics for multi-GPU           available incrementally during computation. We will provide ex-
communication, Iris eliminates the need to learn a new memory            amples that leverage both value-based and pointer-based APIs in
model.                                                                   Section 4.
Iris: First-Class Multi-GPU Programming Experience in Triton


3.2      Host-Side APIs                                                                  1   @triton.jit
                                                                                         2   def load(pointer, to_rank, from_rank, heap_bases, mask=None):
Iris provides Pythonic PyTorch-like host APIs organized into several                     3       translated_ptr = __translate(pointer, to_rank, from_rank,
categories: initialization, memory management, rank/world queries,                               ↩→ heap_bases)

host-side communication, and tensor construction. Iris implements                        4       result = tl.load(translated_ptr, mask=mask)
                                                                                         5       return result
a full symmetric heap, where each allocation returns a PyTorch                           6
tensor that wraps the allocated virtual memory address range. We                         7   @triton.jit
                                                                                         8   def __translate(ptr, from_rank, to_rank, heap_bases):
organize the discussion into three main areas: core infrastructure                       9       from_base = tl.load(heap_bases + from_rank)
setup, distributed operations, and tensor management. Table 2 at                        10       to_base = tl.load(heap_bases + to_rank)
the end provides a quick reference summary of all host-side APIs.                       11       ptr_int = tl.cast(ptr, tl.uint64)
                                                                                        12       offset = ptr_int - from_base
                                                                                        13       to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
3.2.1    Core Infrastructure.                                                           14       translated_ptr_byte = to_base_byte + offset
                                                                                        15       translated_ptr = tl.cast(translated_ptr_byte, ptr.dtype)
   Constructor and Initialization. The initialization process follows 16                         return translated_ptr
several key steps: (1) PyTorch Distributed initialization and rank
assignment, (2) GPU device selection based on rank, (3) symmetric
heap initialization on the selected device, (4) IPC handle creation                          Listing 1: Iris load and pointer translation implementation.
and exchange across all ranks using PyTorch Distributed all-gather
operations, (5) opening of remote IPC handles to establish cross-
GPU memory access, and (6) creation of a tensor containing all
heap base addresses for device-side translation. This setup enables                                 • Range functions: arange and linspace for generating se-
seamless remote memory access through the translate function,                                         quences
which converts local pointers to remote addresses by computing                                      • Random functions: rand, randn, randint, uniform for sam-
offsets and applying them to destination heap bases. Iris sets the                                    pling from various distributions
GPU device and treats each single GPU as its own rank in the                                    All functions support standard PyTorch parameters (dtype, device,
distributed communicator2 .                                                                  requires_grad), providing drop-in compatibility with existing Py-
3.2.2    Distributed Operations.                                                             Torch code. The key innovation is that all allocated tensors reside
                                                                                             in the symmetric heap, enabling direct remote GPU access through
   Rank and Device Queries. Iris provides several query functions for                        Iris’s device-side APIs.
distributed computing context. The get_rank() method returns
the current process’s rank ID in the distributed communicator,
while get_num_ranks() returns the total number of ranks (world                                            Table 2: Iris Host-Side API Summary.
size). The get_heap_bases() method returns a tensor containing
symmetric heap base addresses for all ranks, which is essential for                          Category     Function            Description
device-side pointer translation.                                                                          __init__()          Initialize Iris runtime, setup symmetric heap,
                                                                                                                              exchange IPC handles
   Host-side Communication. Iris provides two primary communi-                                            barrier()           Synchronize all ranks (GPU sync + dis-
                                                                                             Core
cation primitives. The barrier() function synchronizes all ranks                                                              tributed barrier)
                                                                                                          broadcast()
across the entire system by first calling torch.cuda.synchronize()                                                            Broadcast tensor or scalar from one rank to
                                                                                                                              all others
(or stream.synchronize() if a stream is specified) to ensure the                                          get_rank()          Return current process rank ID
local GPU has finished all queued work, then performing a global                                          get_num_ranks()     Return total number of ranks (world size)
                                                                                                          get_heap_bases()    Return tensor containing all symmetric heap
distributed barrier so all ranks reach the same point before pro-                                                             base addresses
ceeding. The broadcast() function broadcasts a value from one
                                                                                                          zeros()             Create tensor filled with zeros in symmetric
rank to all others, automatically detecting the value type and using                                                          heap
the appropriate mechanism: for tensors and arrays it uses efficient                                       ones()              Create tensor filled with ones in symmetric
                                                                                                                              heap
PyTorch distributed tensor collectives, while for scalars and other                          Tensor       empty()             Create uninitialized tensor in symmetric
objects it uses object broadcast. This intelligent type detection                            Creation                         heap
makes broadcasting seamless across different data types.                                                  full()              Create tensor filled with specified value
                                                                                                          zeros_like()        Create zeros tensor matching input tensor’s
3.2.3    Tensor Management.                                                                                                   shape
                                                                                                          rand()              Create tensor with uniform random values
                                                                                                                              [0, 1)
   Tensor Construction. Since remote memory operations require                                            randn()             Create tensor with normal distribution
symmetric heap allocation, Iris provides PyTorch-compatible func-                                                             (mean=0, std=1)
tions for tensor creation and initialization. The library supports                                        randint()           Create tensor with random integers in range
                                                                                                          uniform()           Create tensor with uniform random values
three categories of functions (see Table 2 for the complete list):                                                            in [low, high)
        • Creation functions: zeros, ones, empty, full, zeros_like                                        arange()            Create 1D tensor with evenly spaced values
                                                                                             Sequences
          for basic tensor initialization                                                                 linspace()          Create 1D tensor with linearly spaced values

2 To simplify distributed programming, typically, each GPU is treated as its own rank
rather than a single compute node.
                                                                                                                                                        Awad, Osama and Potter


              Table 3: Iris Device-Side API Summary                            3.3.2 Memory Operations. Iris provides both value-based and pointer-
                                                                               based memory operations (see Table 3):
Category      Function           Description                                        • Value-based operations: load() and store() move data
              load()             Load value from remote rank’s memory                  directly between registers and remote memory, enabling
                                 (value-based)
Memory
              store()            Store value to remote rank’s memory (value-
                                                                                       fine-grained register-to-memory transfers
Operations
                                 based)                                             • Pointer-based operations: get(), put(), and copy() per-
              get()              Copy from remote memory to local memory               form bulk transfers between memory regions, operating on
                                 (pointer-based)
              put()              Copy from local memory to remote memory               buffer-to-buffer copies
                                 (pointer-based)                                  All memory operations are non-blocking and use relaxed mem-
              copy()             Copy between any two ranks (pointer-based)
                                                                               ory ordering by default. The choice between value-based and pointer-
              atomic_add()       Atomically add value to remote memory lo-     based operations depends on the communication pattern and data
                                 cation
              atomic_xchg()      Atomically swap value with remote memory      granularity requirements (discussed further in Section 4).
                                 location
Atomics                                                                        3.3.3 Atomic Operations and Memory Model. Iris provides the com-
              atomic_cas()       Atomically compare and conditionally swap
                                 if values match                               plete HIP/CUDA memory model semantics for atomic operations.
              atomic_and()       Atomically perform bitwise AND on remote
                                 memory
                                                                               The library supports three synchronization scopes:
              atomic_or()        Atomically perform bitwise OR on remote             • block: synchronization visible within a thread block (CTA)
                                 memory
              atomic_xor()       Atomically perform bitwise XOR on remote
                                                                                     • gpu: synchronization visible across the entire GPU
                                 memory                                              • sys: synchronization visible system-wide across all GPUs
              atomic_min()       Atomically compute minimum with remote
                                 memory
                                                                                  Memory ordering options include relaxed, acquire, release,
              atomic_max()       Atomically compute maximum with remote        and acq_rel, enabling fine-grained control over synchronization
                                 memory                                        semantics. The atomic operations include arithmetic (add), bitwise
Translation   __translate()      Internal function for pointer translation     (and, or, xor), comparison (min, max), and exchange (xchg, cas) op-
                                                                               erations. All atomics follow the same pattern: translate the pointer,
Note: All functions require heap_bases parameter. Atomics support sem          then perform the atomic operation with specified memory ordering
      (relaxed/acquire/release/acq_rel) and scope (block/gpu/sys).
                                                                               and scope.
                                                                               3.3.4 Gluon Backend. Iris also provides a Gluon backend that uses
                                                                               Triton’s @gluon.jit decorator and @aggregate to encapsulate
3.3       Device-Side APIs                                                     backend state, eliminating the need to pass heap_bases manu-
Iris provides Pythonic Triton-style device APIs for remote memory              ally. This backend offers improved ergonomics by encapsulating
access and atomic operations. Since Triton doesn’t support object-             the Iris device context in an aggregate type. For example, instead of
oriented programming, all functions require passing the symmetric              passing heap_bases explicitly (standard API: iris.load(buffer,
heap pointer obtained via the get_heap_bases API. All device-                  to_rank, from_rank, heap_bases)), the Gluon API encapsu-
side operations follow a consistent two-step pattern: (1) pointer              lates this in a context object (ctx.load(buffer, from_rank=1)).
translation from local to remote address space, and (2) memory                 However, we focus this paper on the standard Triton API for clarity
operation on the translated pointer. Table 3 summarizes all device-            and broader compatibility.
side APIs.

3.3.1 Pointer Translation Mechanism. The core of Iris’s remote
                                                                                      Virtual Address Space




                                                                                                                              Virtual Address Space




memory access is the __translate function, which enables seam-
less access to remote memory without requiring explicit memory
management from the programmer. The translation process (illus-
trated in Figure 2) works as follows:                                                                               SymHeap
                                                                                                                     Offset                            SymHeap
      (1) Compute the offset of the pointer within the local rank’s                                                 (0x448)                             Offset
          symmetric heap                                                                                                                               (0x448)
                                                                                                               0xFFFCABC0
      (2) Add this offset to the target rank’s heap base address
                                                                                                                Heap Base                             0xFFFC0420
      (3) Cast the result back to the appropriate pointer type
                                                                                                                                                       Heap Base
   In the current version of Iris supporting intra-node communi-
                                                                                                              0x0
cation via IPC, the remote operation can be directly performed on
the computed remote pointer using standard memory operations
(e.g., tl.load). While the implementation loads the heap_bases on
each call, experimental results show that these loads have no over-
head, likely because the heap bases array (64 bytes) remains cached            Figure 2: Pointer translation mechanism in Iris showing how
at the L1 level. Listing 1 shows the implementation of load and                local pointers are converted to remote addresses through
__translate, illustrating the two-step translation-then-operation              offset computation.
pattern.

                                                                        12 |
Iris: First-Class Multi-GPU Programming Experience in Triton


3.4     Iris Features and More                                          1   @triton.jit
                                                                        2   def gemm_loop(A, B, C, ...):
Tile-based APIs and programming model for communication. 3                      # Tile coordinate calculation removed for brevity
Iris provides a tile-based communication model where operations 4               # Memory layout setup
                                                                                rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
are organized into tiles (e.g., BLOCK_SIZE_M, BLOCK_SIZE_N) that 56             rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
naturally fit within cache hierarchies. This tile-based approach en- 7          rk = tl.arange(0, BLOCK_SIZE_K)
ables building larger tiles within L1, L2, and LLC caches on chiplet- 8         A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
                                                                        9       B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
based architectures like MI300X and MI350X. Communication oper- 10
ations work at tile granularity, allowing fine-grained overlap where 11          # Initialize accumulator registers
tiles can be communicated as soon as they are produced, rather 12                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                                                                       13
than waiting for entire computation phases to complete. The tile- 14             # GEMM's Main loop
based model seamlessly integrates with Triton’s blocked tensor 15                for k in range( tl.cdiv(K, BLOCK_SIZE_K)):
                                                                                     a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
operations, providing a unified programming model for both local 16    17            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
computation and remote communication.                                  18            acc += tl.dot(a, b)
    Ease of instrumentation and profiling granularity within 19                      A_BASE += BLOCK_SIZE_K * stride_ak
                                                                       20            B_BASE += BLOCK_SIZE_K * stride_bk
the kernel and library. Since Iris is implemented entirely in Tri- 21
ton, profiling and instrumentation (e.g., through Triton’s official 22           # Non-even K handling removed for brevity
profiler Proton) can be performed at fine-grained granularity both 23            ...
                                                                       24
within user kernels and inside the Iris library itself. Developers can 25        # Accumulator registers with C results
instrument specific communication operations, measure overlap 26                 return acc.to(C.type.element_ty)
efficiency, and analyze performance at the workgroup or even in-
struction level—not just at the boundaries of library calls, but deep       Listing 2: A Triton GEMM main-loop routine, repurposed for
within the library’s implementation. This enables debugging and             several algorithms explained in the paper.
performance analysis of the pointer translation mechanism, remote
memory operations, and synchronization primitives. This contrasts
sharply with wrapped libraries where only opaque function call
boundaries are visible, making it impossible to understand the per-                                           Patterns
formance characteristics of individual communication operations
within a fused kernel or to diagnose issues inside the library itself.
    L1-, L2-, LLC-cache aware programming using swizzling                                 Unfused                                  Fused
and cache modifiers. Iris provides explicit control over cache
behavior through two complementary mechanisms. First, cache                      Bulk-             Producer-                               Producer-
                                                                                                                      Sequential
modifiers on load/store operations (e.g., cache_modifier=".wt"                Synchronous          Consumer                                Consumer
for write-through) allow direct control over how data is written to
memory hierarchy, enabling optimization for chiplet architectures                                                               Workgroup-         Wavefront-
                                                                                                      Work Queue
where cache coherence across XCDs (Accelerator Complex Die)                                                                     Specialized        Specialized

is critical. Second, swizzling is implemented at multiple levels:
(1) across XCDs using chiplet_swizzle to map work-groups to                                                                     Work Queue
specific XCDs grouping tiles together for better Last-Level Cache
(LLC) locality; and (2) spatial swizzling using GROUP_SIZE_M for
L2-cache locality within tiles. This multi-level swizzling strategy,        Figure 3: Taxonomy of unfused and fused computation and
combined with cache modifiers, allows building larger tiles that            communication overlap patterns.
efficiently utilize the entire cache hierarchy, from L1 caches within
individual compute units to the LLC shared across XCDs.
                                                                            must be rapidly exchanged across devices (via all-scatter) to form
4     Building Complex Multi-GPU Patterns                                   the complete global output.
                                                                               With Iris, these patterns can be expressed naturally within Tri-
Iris enables sophisticated distributed algorithms through a simple          ton, developers can write kernels that overlap GEMM computation
yet powerful API. To illustrate this, we present a taxonomy of              with communication, eliminating execution “bubbles”3 . This over-
fused and unfused compute-communication patterns, using General             lap is notoriously difficult to achieve in practice, yet Iris makes
Matrix Multiplication (GEMM) and all-scatter as a case study.               it straightforward through intuitive APIs that integrate remote
   GEMM is the foundational building block of modern GPU work-              memory operations directly into the Triton programming model.
loads, from deep learning to scientific simulation, responsible for            In contrast to the traditional complexity of building fused kernels,
most of the floating-point operations in large models. All-scatter,         Iris enables developers to construct multi-GPU pipelines that are
on the other hand, is a collective communication primitive where            both efficient and maintainable. As we show next, this provides a
each GPU (or rank) distributes distinct portions of its data to all
other GPUs. Together, they represent a common and challenging               3 Bubbles refer to idle pipeline stages where no useful work occurs, often due to kernel
pattern: local compute producing partial results (via GEMM) that            launch overhead or synchronization delays.
                                                                                                                                  Awad, Osama and Potter


practical taxonomy of compute-communication overlap strategies 1               @triton.jit()
                                                                    2          def gemm(
captured in Figure 3.                                               3              A, B, C, ...
   We organize the patterns into two main categories: unfused pat- 4           ):
terns where computation and communication execute in separate 5                    pid = tl.program_id(0)
                                                                    6              for tile_id in range(pid, total_tiles, NUM_SMS):
kernels, and fused patterns where both operations are combined 7                       c = gemm_loop(A, B, C)
within a single kernel. Each category offers different trade-offs 8                    ...
between implementation complexity, resource utilization, and per- 9                    # Store to local GPU's memory
                                                                   10                  tl.store(C + offset, c, mask=mask, cache_modifier=".wt")
formance characteristics.                                          11
                                                                          12   @triton.jit()
                                                                          13   def all_scatter(C, ...):
4.1    Unfused Patterns                                                   14       pid = tl.program_id(0)
                                                                 15                for tile_id in range(pid, total_tiles, NUM_SMS):
Unfused patterns separate computation and communication into 16
distinct kernels, providing clear boundaries between operations. 17                    # Begin: See the if segment for explanation:
We begin with the simplest approach and progress to more sophis- 18                    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
                                                                 19                    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
ticated producer-consumer strategies.                            20                    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M),
                                                                                       ↩→ BLOCK_SIZE_M)
4.1.1 Bulk-Synchronous. The simplest approach to coordinating 21                       rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N),
compute and communication is the bulk-synchronous pattern, where                       ↩→ BLOCK_SIZE_N)
                                                                       22              mask = (rm[:, None] < M) & (rn[None, :] < N)
operations execute sequentially with explicit synchronization bar- 23                  offset = rm[:, None] * stride_cm_global + (rn[None, :] +
riers between kernels. In this pattern, the GEMM kernel first com-                     ↩→ cur_rank * N) * stride_cn_global
pletes all computation and stores results to local GPU memory, 24                      # End: masks/offset calculations.
and only after the entire GEMM kernel finishes and synchronizes 25     26              # Store from local to all other GPU's memory
does the all-scatter kernel begin, reading from local memory and 27                    for remote_rank in range(world_size):
distributing data to remote GPUs using iris.put. Listing 3 demon- 28                       if remote_rank != cur_rank:
                                                                       29                      iris.put(C + offset, C + offset,
strates this pattern: two separate kernels are launched sequentially 30                        cur_rank, remote_rank, heap_bases, mask=mask)
on the same stream, establishing a strict data dependency that en- 31
sures the GEMM kernel fully completes before any communication 32              # On a single stream launch both kernels,
                                                                       33      # establishing dependency of the two operations.
begins.                                                                34      with torch.cuda.stream(main_stream):
   This approach offers the benefit of simplicity and clear separation 35          gemm[(num_sms,)](A, B, C, ...)
                                                                               with torch.cuda.stream(main_stream):
of concerns—each kernel has a single, well-defined responsibility. 36  37          all_scatter[(num_sms,)](C, ...)
However, it introduces significant execution “bubbles” as shown in
Figure 4: the GPU must wait for all GEMM work to complete and all
workgroups to synchronize before any communication can proceed,                Listing 3: Iris: Unfused, Bulk Synchronous – illustrates the
leaving computational resources idle during the synchronization                use of iris.put in a separate kernel after the GEMM kernel
barrier. The pattern also requires intermediate writes to global               concludes and synchronizes.
memory, as the GEMM results must be stored before the all-scatter
kernel can read them, adding memory bandwidth overhead that
could be avoided with more sophisticated overlap strategies.

4.1.2 Producer-Consumer (Stream Concurrency). Building on the                  48 SMs to perform all-scatter communication. The producer kernel
bulk-synchronous pattern, the unfused producer-consumer ap-                    writes completed tiles to local memory and signals their availability
proach achieves overlap by launching two separate kernels on                   using atomic operations (e.g., tl.atomic_cas with release seman-
asynchronous streams with explicit resource partitioning. Unlike               tics). The consumer kernel, executing concurrently on a separate
bulk-synchronous execution where each kernel uses the entire                   stream, spins on these atomic locks (with acquire semantics) and
GPU sequentially, this pattern limits the number of compute units              immediately begins scattering tiles to remote GPUs as they become
(CUs) or streaming multiprocessors (SMs) allocated to each kernel.             available. This pattern offers the modularity benefits of separate
One kernel (the producer) uses a subset of CUs to perform GEMM                 kernels while achieving overlap through hardware concurrency,
computation, while another kernel (the consumer) uses the remain-              though it requires careful tuning of the CU partition to balance
ing CUs to perform communication. Dependencies between the                     computation and communication workloads.
kernels are managed through atomic-based synchronization primi-
tives (similar to Listing 5), but instead of using an if/else statement
within a single fused kernel, the two operations execute as separate
                                                                               4.2    Fused Patterns
kernels on different streams. This approach enables concurrent                 While unfused patterns provide simplicity and modularity, fused
execution of computation and communication while maintaining                   patterns offer superior performance by eliminating kernel launch
explicit control over resource allocation.                                     overhead and enabling fine-grained computation-communication
   For example, on an MI300X with 304 compute units, the pro-                  overlap within a single kernel. These patterns leverage Iris’s native
ducer kernel might be launched with 256 SMs to handle GEMM tiles,              Triton implementation to seamlessly interleave operations at tile
while the consumer kernel runs concurrently with the remaining                 granularity.
Iris: First-Class Multi-GPU Programming Experience in Triton


                                   GPU0’s View — Bulk-Synchronous                           GPU0’s View — Producer-Consumer
Kernel Launch
      Latency

         GEMM          CU0                                                        CU0
         Tile

Communication          CU1                                                        CU1
         Tile


  Synchronize   🔒      CU2                                                        CU2

                       CU3                                                        CU3            🔒0             🔒          🔒


                       CU4                                                        CU4            🔒1              🔒3            🔒



                                                        Time                                                   Time


Figure 4: Timeline: Illustrates a single GPU’s view of the taxonomy of unfused computation and communication patterns.
(left) bulk-synchronous highlights the hard synchronization barriers that exist after each kernel, and (right) a multi-kernel
producer-consumer pattern shows how overlap can be achieved by moving the synchronization at a finer granularity and
partitioning the compute units (CUs) between computation and communication workers.

                                         GPU0’s View — Sequential                        GPU0’s View — Workgroup Specialization
Kernel Launch
      Latency

   Fused GEMM
                        CU0                                                        CU0
         Tile

        Fused           CU1                                                        CU1
Communication
         Tile

  Synchronize   🔒       CU2                                                        CU2

                        CU3                                                        CU3           🔒0              🔒         🔒


                                                                                   CU4           🔒1                 🔒3         🔒
                        CU4

                                                         Time                                                   Time


           Figure 5: Timeline: Illustrates a single GPU’s view of the taxonomy of fused GEMM and All-Scatter patterns.


4.2.1 Sequential. To bring more control of scheduling the work             the next operator in the same kernel. The impact of tail latency
in the hands of developers, we can fuse multiple operations to-            (tail occupancy inefficiency) worsens, because now the last “wave”
gether in a mega or uber kernel [8, 23] and reduce the overhead            of work needs to process GEMM and all scatter before the kernel
of tearing down and recreating the kernels. This approach sig-             completes.
nificantly reduces the “bubbles” in the total workload by moving
to a fine-grained synchronization approach at a tile granularity.          4.2.2 Producer-Consumer (Workgroup Specialization). One such
One such way fused kernels are implemented is following the                way of avoiding the sequential issuance of the two operator is by us-
data-dependencies that inherently exist within those operators,            ing specialization techniques over the available compute resources.
for example, we insert the all-scatter operator sequentially after the     We can implement a persistent-style kernels (see the for-loop over
computation of each output GEMM tile. Iris enables this pattern            all tiles in Listings 4) and specialize the type of computation each
through its device-side APIs that operate directly within Triton           compute resource (e.g., workgroups) does by using the workgroup
kernels. Listing 4 illustrates how developers may use iris.store           index (pid in Triton). Listing 5 shows an example of GEMM + All
to immediately scatter the GEMM tile produced to all remote GPUs           Scatter using Iris, where each workgroup gets mapped to a compute
without the need for a bulk-synchronous kernel-level barrier (see          unit of an AMD GPU (MI300X has 304 compute units); the first 256
Listing 3).                                                                (0-255) workgroups are responsible for computing the GEMM out-
    Such pattern has the benefit of operating on the data as soon as it    put tile and signaling the other 48 workgroups (256-303) responsible
is ready (such as all scatter’s store on the accumulator registers) with   for waiting for a tile to be produced (using for example a spin-lock),
no intermediate writes to global memory required. Fused operators,         and then scatter the result to other GPUs. With this method, we can
however, still retain the sequential dependencies of executing one         dedicate exact compute resources for various tasks—this is espe-
operator, waiting for example, the GEMM to complete and issuing            cially useful when workloads like GEMMs do not require the entire
                                                                                                                                     Awad, Osama and Potter


1    @triton.jit                                                              1   @triton.jit()
2    def fused_gemm_all_scatter(                                              2   def wg_specialized_gemm_all_scatter(
3        A, B, C, ...                                                         3       A, B, C, locks, GEMM_SMS, COMM_SMS, ...
4    ):                                                                       4   ):
5        pid = tl.program_id(0)                                               5       pid = tl.program_id(0)
6        num_pid_m, num_pid_n = tl.cdiv(M, BLOCK_SIZE_M), tl.cdiv(N,          6       if pid < GEMM_SMS:
         ↩→ BLOCK_SIZE_N)                                                     7
7        total_tiles = num_pid_m * num_pid_n                                  8             # Process all gemm tiles using GEMM_SMS number of
8        for tile_id in range(pid, total_tiles, NUM_SMS):                     9             # workgroups in a persistent fashion.
9            c = gemm_loop(A, B, C)                                          10             for tile_id in range(pid, total_tiles, GEMM_SMS):
10                                                                           11                 c = gemm_loop(A, B, C)
11           rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M    12                 ...
12           rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N    13                 # Store to local GPU's memory
13                                                                           14                 tl.store(C + offset, c, mask=mask, cache_modifier=".wt")
14           # Add compiler hints                                            15                 tl.atomic_cas(locks + tile_id, 0, 1, sem="release",
15           rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M),                           ↩→ scope="gpu")
             ↩→ BLOCK_SIZE_M)                                                16
16           rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N),        17         else: # pid >= GEMM_SMS
             ↩→ BLOCK_SIZE_N)                                                18             COMM_SMS = NUM_SMS - GEMM_SMS
17                                                                           19             pid = pid - GEMM_SMS
18           # Define the C-mask (BLOCK_SIZE_M, 1) x (1, BLOCK_SIZE_N)       20

19           mask = (rm[:, None] < M) & (rn[None, :] < N)                    21             # Process all comm tiles using COMM_SMS number of
20                                                                           22             # workgroups in a persistent fashion.
21           # Calculate the "global" offset of C based on the rank.         23             for tile_id in range(pid, total_tiles, COMM_SMS):
22           # Note the N-dimension is being multiplied by current rank.     24

23           # This is because each rank is computing a portion of the       25                # Wait for the tile to be ready.
24           # N-dimension locally and then scattering it to all other       26                while tl.atomic_cas(locks + tile_id, 1, 0, sem="acquire",
25           # ranks to complete the global N-dimension.                                       ↩→ scope="gpu") == 0:
26           offset = rm[:, None] * stride_cm_global + (rn[None, :] +        27                       pass
             ↩→ cur_rank * N) * stride_cn_global                             28

27                                                                           29                # Store from local to all other GPU's memory
28           # Scatter to all ranks                                          30                for remote_rank in range(world_size):
29           for remote_rank in range(world_size):                           31                    if remote_rank != cur_rank:
30               iris.store(C + offset, c, cur_rank, remote_rank,            32                        iris.put(C + offset, C + offset,
                 ↩→ heap_bases, mask=mask)
                                                                             33                        cur_rank, remote_rank, heap_bases, mask=mask)
                                                                             34
                                                                             35   # Launch code:
                                                                             36   with torch.cuda.stream(main_stream):
     Listing 4: Iris: Fused, Sequential – illustrates the use of             37       wg_specialized_gemm_all_scatter[(num_sms,)](
     iris.store right after the GEMM tile is produced.                       38           A, B, C, locks, GEMM_SMS, COMM_SMS, ...)



     device to achieve peak performance. Fused workgroup specializa-              Listing 5: Iris: Fused, Workgroup Specialization – illustrates
     tion, however, just like all other fused kernels, requires worst-case        how a single fused kernel can be split into components where
     resource allocation (i.e., an operation such as all-scatter is forced        dedicated workgroups perform either communication or
     to occupy more resources than needed because it is fused with a              computation operations.
     more resource-intensive operation such as GEMM).
     4.2.3 Producer-Consumer (Wave Specialization) and Work Queue.
     Similar to workgroup specialization, we can also split the work at           primitives, and real-world application studies using the GEMM+All-
     a finer granularity of a wavefront (AMD GPU) or a warp (NVIDIA               Scatter fused patterns from Section 4. The microbenchmarks demon-
     GPU), where 64 or 32 threads in a lockstep fashion work on issuing           strate that Iris achieves near-optimal bandwidth utilization across
     communication or processing compute. However, without using                  all operations, validating the efficiency of its native Triton imple-
     Gluon, this pattern is not typically suited for a more workgroup-            mentation. For real-world workloads, Iris’s fine-grained overlap
     centric language like Triton. Work queue on the other hand extends           capabilities enable significant performance improvements over the
     these patterns and moves the management of the work in a separate            bulk-synchronous baseline, with speedups ranging from 0.93× to
     queue-like data structure, due to the synchronization cost of in-            1.79× (average 1.21×) compared to PyTorch and RCCL. These re-
     serting and removing “work” (communication or computation tile),             sults highlight both the low overhead of Iris’s abstraction and the
     queues are also not well suited for a GPU architecture (or Triton            substantial benefits of tile-granularity computation-communication
     language.) In this paper, we focus on all other patterns described in        overlap enabled by its design.
     the previous subsections.
                                                                                  5.1      Microbenchmarks
     5   Results                                                                  Figure 6 presents performance benchmarks for point-to-point load,
     We evaluate Iris on a system with 8x AMD Instinct™ MI300X                    store, and atomic operations. All benchmarks are normalized rela-
     GPUs configured under NPS1/SPX memory and compute partition                  tive to the achievable bandwidth [21, 22] on the system, with the
     modes [19] and ROCm 6.3.1. Our evaluation consists of two main               heatmaps showing bandwidth percentages where darker shades
     components: microbenchmarks that characterize Iris’s fundamental             indicate better (higher) performance. The results demonstrate Iris’s
     performance across point-to-point and collective communication               efficiency in handling different types of remote memory access
  Iris: First-Class Multi-GPU Programming Experience in Triton


                                                                         100                                                                                                      100
             93.3   98.7   99.0     97.9   97.7     99.1   97.5   96.3                                                99.8   98.7   98.1     96.5   97.8     98.1   96.5   95.9
      00




                                                                                                               00
             99.1   96.2   98.4     99.0   98.4     97.9   97.6   96.9   80                                           98.9   99.3   97.5     98.2   97.6     97.7   96.4   96.2   80
      01




                                                                                                               01
             98.7   98.3   99.2     98.2   96.7     96.8   96.0   96.4                                                98.3   97.5   99.2     98.1   95.4     96.3   94.9   95.1




                                                                              Normalized Bandwidth (%)




                                                                                                                                                                                       Normalized Bandwidth (%)
      02




                                                                                                               02
                                                                         60                                                                                                       60
Source GPU




                                                                                                         Source GPU
             98.1   98.9   99.0     96.5   97.7     97.3   95.5   96.0                                                97.9   98.6   98.0     99.8   96.5     96.6   95.9   95.6
      03




                                                                                                               03
             98.0   98.6   96.6     97.1   99.0     99.2   98.8   98.5                                                97.6   97.8   96.0     97.5   98.6     98.5   98.0   98.4
      04




                                                                                                               04
                                                                         40                                                                                                       40
             98.6   97.8   97.3     97.2   99.2     99.9   98.4   98.5                                                97.7   97.8   96.8     96.8   98.4     99.4   98.5   98.4
      05




                                                                                                               05
             97.7   97.5   95.7     95.9   98.8     99.4   95.2   98.4   20                                           96.5   96.0   95.3     95.4   97.6     97.6   99.2   97.5   20
      06




                                                                                                               06
             96.3   97.1   96.1     96.0   99.0     99.1   98.4   97.2                                                95.7   96.6   95.8     95.4   98.5     98.5   98.0   99.7
      07




                                                                                                               07
             00     01     02       03      04      05     06     07     0                                            00     01     02       03      04      05     06     07     0
                                  Destination GPU                                                                                          Destination GPU
                                  (a) Load benchmark                                                                                       (b) Store benchmark
                                                                         100                                                                                                      100
             98.6   98.7   99.1     97.4   98.6     99.2   96.8   95.8                                                98.4   99.3   99.5     99.5   99.4     99.6   99.5   99.5
      00




             99.2   99.2   98.5     99.7   99.2     98.6   97.5   97.5   80                                    00     99.6   99.5   99.6     99.6   99.5     99.5   99.6   99.5   80
      01




                                                                                                               01

             99.2   98.9   99.5     98.6   96.2     97.5   95.9   95.9                                                99.5   99.5   98.9     99.5   99.6     99.6   99.6   99.6
                                                                              Normalized Bandwidth (%)




                                                                                                                                                                                       Normalized Bandwidth (%)
      02




                                                                                                               02



                                                                         60                                                                                                       60
Source GPU




                                                                                                         Source GPU




             97.8   99.6   98.6     99.0   96.9     97.4   95.9   95.3                                                99.6   99.6   99.5     98.8   99.6     99.5   99.5   99.6
      03




                                                                                                               03




             98.0   98.6   95.9     97.4   98.8     99.1   98.1   99.1                                                99.6   99.5   99.5     99.6   99.2     99.6   99.6   99.6
      04




                                                                                                               04




                                                                         40                                                                                                       40
             98.6   98.6   97.5     98.0   99.0     99.8   99.2   98.6                                                99.6   99.6   99.6     99.5   99.6     98.8   99.6   99.6
      05




                                                                                                               05




             97.4   97.5   95.9     96.3   98.1     99.2   98.4   98.6   20                                           99.6   99.5   99.6     99.6   99.5     99.5   98.0   99.6   20
      06




                                                                                                               06




             96.4   96.5   96.3     95.9   98.0     98.6   98.0   99.3                                                99.6   99.5   99.5     99.6   99.6     99.6   99.6   99.8
      07




                                                                                                               07




             00     01     02       03      04      05     06     07     0                                            00     01     02       03      04      05     06     07     0
                                  Destination GPU                                                                                          Destination GPU
                                     (c) Atomic add                                                                                          (d) Atomic xchg

  Figure 6: Performance benchmarks for load, store, and atomic operations. The results demonstrate Iris’s efficiency in handling
  different types of remote memory access patterns.


  patterns, with consistent performance across operations, achieving                                        5.2        Evaluating Fused, Unfused Patterns
  near-optimal bandwidth utilization.                                                                                  Taxonomy
     Figure 7 provides the all-load and all-store mircobenchmark re-
                                                                                                            To evaluate Iris’ in a real-world application, we continue our case-
  sults where all GPUs participate in the load and store operations
                                                                                                            study from Section 4. We implemented many of the fused and
  across all links. In the benchmark, different buffer sizes are moved
                                                                                                            unfused patterns described in the Figure 3 and Listings 3, 4 and 5 to
  across all ranks at the same time. The heatmaps show the nor-
                                                                                                            capture the versatility of the abstraction and APIs. In this section, we
  malized bandwidth relative to achievable bandwidth where darker
                                                                                                            cover a deep-dive of these patterns using PyTorch’s torch.matmul
  shades represent superior performance. As buffer size increases,
                                                                                                            and RCCL’s All-Gather as a functionally equivalent baseline. Fig-
  performance improves significantly (as expected), reaching near-
                                                                                                            ure 8 shows the complete performance landscape across six different
  optimal achievable bandwidth utilization showcasing the efficiency
                                                                                                            problem shapes and sizes with varying N and K-dimensions (and
  of the Iris simple-yet-effective implementation and abstraction.
                                                                                                            M=8192) for different number of GPUs (world size).
                                                                                                               We first establish Iris’ baseline using the “Unfused,
                                                                                                            Bulk-synchronous” schedule, which as a schedule is equivalent
                                                                                                                                                                                  Awad, Osama and Potter


                                                                               100                                                                                                                100
                1MB 70.8    65.6   65.8   65.8   63.0     69.0   64.1   61.8                                                   1MB 22.0    22.5   18.7   19.9   20.0     18.1   18.2   18.6
                2MB 75.1    73.5   73.5   70.7   69.3     70.3   69.2   69.9                                                   2MB 25.3    23.6   20.3   21.7   22.0     19.7   19.2   20.1
                4MB 78.7    80.7   79.5   80.0   77.9     79.7   79.0   78.0   80                                              4MB 30.8    28.3   25.3   29.8   30.7     28.5   26.2   25.8       80
                8MB 84.5    85.7   85.1   85.9   85.2     87.8   88.0   86.0                                                   8MB 42.9    37.5   33.3   39.3   41.6     38.0   35.2   34.4




                                                                                    Normalized Bandwidth (%)




                                                                                                                                                                                                       Normalized Bandwidth (%)
               16MB 83.4    89.2   83.8   87.5   85.1     87.5   90.8   86.3                                                  16MB 55.4    49.9   45.3   49.8   53.5     49.7   46.9   46.4
               32MB 86.7    90.7   87.3   90.8   87.3     90.1   92.4   88.5   60                                             32MB 69.3    64.9   60.9   61.8   66.2     63.1   61.0   61.6       60
Buffer Size




                                                                                                               Buffer Size
               64MB 86.9    94.6   87.3   92.8   89.1     90.4   96.2   89.6                                                  64MB 80.4    78.2   74.1   76.0   78.2     75.9   74.5   75.0
              128MB 87.1    94.8   87.5   94.3   90.8     93.5   98.7   91.8                                                 128MB 88.9    86.6   84.0   85.1   86.9     85.3   83.6   84.3
                                                                               40                                                                                                                 40
              256MB 87.7    95.0   88.1   94.6   90.4     93.7   97.7   91.5                                                 256MB 94.0    92.6   91.1   91.5   92.4     91.6   90.2   90.2
              512MB 87.4    95.6   87.8   95.2   90.7     93.7   99.3   92.1                                                 512MB 97.2    96.3   94.9   94.9   95.5     95.1   94.0   94.2
              1024MB 87.4   95.8   87.9   95.5   90.5     93.7   99.0   92.5   20                                            1024MB 98.7   98.2   96.9   96.7   97.4     97.1   95.8   96.1       20
              2048MB 87.9   95.4   88.5   95.4   90.1     93.5   98.2   91.6                                                 2048MB 99.3   99.0   97.9   97.7   97.8     98.0   96.4   97.1
              4096MB 88.3   96.3   88.9   96.2   90.3     93.8   99.1   91.9                                                 4096MB 98.9   98.8   97.7   97.4   97.7     97.6   96.5   96.8
                      00    01     02     03         04   05     06     07     0                                                     00    01     02     03         04   05     06     07         0
                                               GPU                                                                                                            GPU
                                   (a) All-Load benchmark                                                                                         (b) All-Store benchmark

  Figure 7: All load/all store benchmark results where all GPUs participate in all load/store operations across all links. As buffer
  size increases, performance approaches peak bandwidth utilization as expected.




  Figure 8: Complete performance comparison between Iris fused GEMM + All-Scatter and RCCL GEMM + AllGather kernels
  across different problem sizes and world sizes.


 PyTorch and RCCL. We observe that Iris competes with state-of-the-                                                   Iris also allows to break the rigid bulk-synchronous program-
 art GEMM and All-Gather implementation — this further validates                                                   ming model by using the device-side APIs and moving the synchro-
 that Iris’ abstraction isn’t resulting in any discernible overheads. In                                           nization at a fine grained tile-level granularity. “Unfused, producer-
 some cases, such as 8192 × 4608 × 36864, Iris is 20% faster using 8                                               consumer”, “fused, workgroup-specialization” and “fused, sequen-
 GPUs. This is largely attributed to difference in heuristics for the Py-                                          tial” all follow this model. Unfused, producer-consumer approach
 Torch and RCCL’s heuristics selecting a suboptimal configuration                                                  gives up-to 2.5× speedup for problem shape of 8192 × 3584 × 14336
 (see Figure 9).
Iris: First-Class Multi-GPU Programming Experience in Triton


                                                   8192 × 4608 × 36864                                                                                                                8192 × 3584 × 14336
                               torch+rccl                                iris (unfused, bulk-synchronous)                                                         torch+rccl                                 iris (unfused, producer-consumer)
             8
                                                                                                            GEMM                             2.5                                                                                                     GEMM
                 0.78                                                                                                                                                                                0.07
             7                                                                                              COMM                                                                                                                                     COMM
                                                                                                            GEMM                                                                                                                                     GEMM
             6                                                                                              COMM                              2                                                                                                      COMM
                                                                                                                                                   0.63                                                             0.08
                                                                  0.07
             5                                                                                                                                            1.38
 Time (ms)




                                                                                                                   Time (ms)




                                                                                                                               Time (ms)




                                                                                                                                                                                                                                                            Time (ms)
                                                                                                                                             1.5
             4          1.59                                                                                                                                                   1.06
                                            1.39                                                                                                                                      1.00           2.27                               0.12
                 6.39                                                                                                                         1
             3                                                                0.99
                                                                  5.00                                                                             1.57                                                             1.65
                                                                                                  1.07                                                                                                                                               0.10
             2                                     1.17
                        3.06                                                                                0.91                             0.5          0.94
                                            2.80                              2.49                                                                                                                                                      0.89
                                                                                                                                                                               0.77
             1
                                                                                                  1.60                                                                                0.63                                                           0.56
                                                   1.39                                                     1.17
             0                                                                                                                                0
                  1      2                   4      8              1            2                  4         8                                       1      2                   4       8             1               2                  4            8
                               World Size                                            World Size                                                                  World Size                                                World Size


                                                                                                                                                                                      8192 × 8192 × 30720
                                                                                                                                                                 torch+rccl                                 iris (fused, workgroup specialization)
Figure 9: Deep-dive: Shows breakdown of GEMM (darker re-                                                                                                                                                                                             GEMM
                                                                                                                                                   0.82                                              0.06
gion) and Communication (lighter region) for Iris compared                                                                                   10
                                                                                                                                                                                                                                                     COMM
                                                                                                                                                                                                                                                     GEMM
to PyTorch and RCCL for 8192 × 4608 × 36864 matrix size. Note                                                                                 8
                                                                                                                                                                                                                                                     COMM


the slight speedups in both GEMM and communication due                                                                                                    2.39




                                                                                                                               Time (ms)




                                                                                                                                                                                                                                                            Time (ms)
                                                                                                                                              6                                                                     0.40
to Iris’ flexibility to be able to cater to a specific problem                                                                                                                                      10.46
                                                                                                                                                   9.47
shape and size. Tile-based abstraction allows for users to                                                                                    4
                                                                                                                                                                               1.76
                                                                                                                                                          5.61                                                                          0.07
simply adjust the needed tile-size at compile-time per a prob-                                                                                2                                       1.29
                                                                                                                                                                                                                    5.05                             0.16
                                                                                                                                                                               2.46                                                     2.57
lem/kernel granularity.                                                                                                                                                               1.35                                                           1.79
                                                                                                                                              0
                                                                                                                                                    1      2                    4      8              1               2                  4            8
                                                                                                                                                                 World Size                                                World Size


on 8 GPUs. The nature of the problem (small-N after being split 8-
ways, and large-K) allows producer-consumer-style model (unfused                                                               Figure 10: Deep-dive: Shows breakdown of GEMM (darker
or fused using workgroup-specialization) to completely hide the                                                                region) and Communication (lighter region) for Iris com-
communication operation behind the GEMM operation, this is illus-                                                              pared to PyTorch and RCCL for 8192 × 3584 × 14336 matrix size.
trated in Figure 10. The difference between the fused and unfused                                                              These two problems are specifically of interest for producer-
variants of producer-consumer approach is that using unfused two                                                               consumer and workgroup specialization based schedules as
kernels (one producer and one consumer), we avoid worst-case                                                                   they show the approach mostly hides the overhead of com-
resource allocation4 (typically bounded by GEMMs) and promotes                                                                 munication behind the GEMM (darker region) by splitting
better occupancy at the cost of kernel launch latency and less con-                                                            the available GPU’s compute units between GEMM and com-
trol over the scheduling of operations. Whereas in a fused variant,                                                            munication.
we only launch one kernel and do not have to pay an additional
cost to launch the kernel, it allows for more scheduling control and
re-purposing/reusing resources and data when relevant, however,
                                                                                                                                                                                      8192 × 4608 × 36864
requires worst-case resource allocation and limits the occupancy                                                                             8
                                                                                                                                                                 torch+rccl                                        iris (fused, sequential)

                                                                                                                                                                                                                                                     GEMM
for one of the operator.                                                                                                                     7
                                                                                                                                                   0.78
                                                                                                                                                                                                                                                     COMM
                                                                                                                                                                                                                                                     GEMM
   A fused, sequential schedule is the simplest of them all — essen-                                                                         6                                                                                                       COMM
                                                                                                                                                                                                     0.06
tially appending the communication operation at the end of the                                                                               5
                                                                                                                                 Time (ms)




                                                                                                                                                                                                                                                            Time (ms)
                                                                                                                                                                                                                    0.21
main-loop of the GEMM operation. This required a few lines of                                                                                4            1.59
                                                                                                                                                                              1.39
                                                                                                                                                   6.39
code changes as shown in Listing 4, and works well for problems                                                                              3
                                                                                                                                                                                                     5.10                               0.31
that need more resources for GEMM (such as small N and massive                                                                               2
                                                                                                                                                          3.06
                                                                                                                                                                                      1.17                          3.80                             0.22
                                                                                                                                                                              2.80
large-K.) However, as the name suggest, this sort of schedule cre-                                                                           1
                                                                                                                                                                                      1.39
                                                                                                                                                                                                                                        2.03
                                                                                                                                                                                                                                                     1.50
                                                                                                                                             0
ates a sequential dependency between the GEMM operation and                                                                                         1      2                    4      8              1              2                   4            8
                                                                                                                                                                 World Size                                                World Size
All-Scatter operation, and increases the tail latency of the entire
problem. Potentially creating large bubbles in the last timestep of
the problem (also illustrated in Figure 5.) With this schedule, Iris                                                           Figure 11: Deep-dive: Shows breakdown of GEMM (darker re-
outperforms the baseline PyTorch and RCCL implementation by                                                                    gion) and Communication (lighter region) for Iris compared
1.8× for 4 GPUs and 1.5× for 8 GPUs on 8192×4608×36864 problem                                                                 to PyTorch and RCCL for 8192 × 4608 × 36864 matrix size. This
size (see Figure 11).                                                                                                          shape illustrates when fused sequential approach benefits
   Across all tested configurations, Iris achieves an average speedup                                                          when the added communication is an overall small overhead
of 1.21× over PyTorch and RCCL, with speedups ranging from 0.93×                                                               (smaller output tile and really large K) and more resources
to 1.79×, highlighting the consistent performance benefits of Iris’s                                                           can simply be allocated to processing GEMM.
design. These speedups are largely attributed to the flexibility of Iris’
design to be able to implement a fused kernel with tile-granularity
synchronization (and the resultant compute and communication
4Worst-case resource allocation: Size of the allocated shared-memory, number of                                                overlap) versus a rigid, bulk-synchronous programming model of
VGPRs or number of threads launched is not bounded by the worst-case operation.                                                PyTorch’s GEMM and RCCL’s AllGather kernels.
                                                                                                                                                Awad, Osama and Potter


6    Conclusion and Future Work                                                          Xin Jin, and Xin Liu. 2024. FLUX: Fast Software-Based Communication Overlap
                                                                                         on GPUs Through Kernel Fusion. https://doi.org/10.48550/arXiv.2406.06858
Iris’s ability to match or exceed the performance of heavily-optimized                   arXiv:2406.06858 [cs.LG]
HIP/CUDA-based libraries like RCCL despite its pure Triton imple-                    [7] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Meghan
                                                                                         Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin,
mentation demonstrates that low-level native implementations are                         and Arvind Krishnamurthy. 2018. TVM: An Automated End-to-End Optimiz-
not a fundamental requirement for multi-GPU programming. The                             ing Compiler for Deep Learning. https://doi.org/10.48550/arXiv.1802.04799
1.79× peak speedup reflects a qualitatively different programming                        arXiv:1802.04799 [cs.LG]
                                                                                     [8] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu,
model where synchronization granularity shifts from kernel bound-                        Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
aries to tiles. Perhaps more significant is what Iris makes tractable:                   Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun
our taxonomy demonstrates that diverse overlap patterns—from                             Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang,
                                                                                         Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian
bulk-synchronous to fused sequential to producer-consumer to                             Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi
workgroup specialization—can be implemented with minimal code                            Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie
                                                                                         Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin
changes, often requiring only a few additional lines within the same                     Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong
Triton kernel. Patterns that would demand substantial engineering                        Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua
effort with traditional CCLs (separate kernel implementations, com-                      Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang,
                                                                                         Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen,
plex host-side coordination, manual resource partitioning) emerge                        R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu
naturally in Iris using the same primitives Triton developers already                    Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen,
use for single-GPU work.                                                                 Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping
                                                                                         Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L.
    This suggests that the real barrier to fine-grained overlap has                      Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun
been abstraction mismatch, not hardware capability. When commu-                          Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi,
                                                                                         Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang,
nication primitives live in the same semantic space as computation                       Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin
(tile-based Triton), overlap patterns become straightforward exten-                      Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou,
sions rather than heroic engineering efforts. Future work will focus                     Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X.
                                                                                         Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao,
on extending Iris to multi-node settings with RDMA, exploring                            Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi,
additional fused patterns such as wave-specialization in Gluon and                       Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang
work queues, and investigating opportunities to offload Iris opera-                      Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang,
                                                                                         Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting
tions to the compiler itself, leveraging Triton’s ability to optimize                    Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren,
across the entire computation-communication pipeline.                                    Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda
                                                                                         Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan,
                                                                                         Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu,
Acknowledgments                                                                          Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan.
                                                                                         2025. DeepSeek-V3 Technical Report. https://doi.org/10.48550/arXiv.2412.19437
This work was supported in part by Advanced Micro Devices, Inc.                          arXiv:2412.19437 [cs.CL]
under the AMD AI & HPC Cluster Program. The authors would                            [9] HSA Foundation. 2018. HSA Platform System Architecture Specification. https:
                                                                                         //hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf. [On-
like to thank Karl Schulz, Lei Zhang, Ziyad AlBanoby, Octavian-                          line; accessed 2-November-2025].
Alexandru Trifan, David Sidler, Xiaohu Guo, Karthik Sangaiah,                       [10] Andrew Kerr, Duane Merrill, Julien Demouth, and John Tran. 2017. CUTLASS:
Lixun Zhang, Vinayak Gokhale, Panagiotis Mylonas, Eric Eaton,                            Fast Linear Algebra in CUDA C++. https://devblogs.nvidia.com/cutlass-linear-
                                                                                         algebra-cuda/
Aditya Nandakumar, Ahmed Eltantawy, Dimple Prajapati, Mike                          [11] Wanchao Liang, Tianyu Liu, Less Wright, Will Constable, Andrew Gu, Chien-
Chu, Mike Schulte, Ganesh Dasika, Brad Beckmann, Ralph Wittig,                           Chin Huang, Iris Zhang, Wei Feng, Howard Huang, Junjie Wang, Sanket Pu-
and Peng Sun for their continuous feedback, support and sugges-                          randare, Gokul Nadathur, and Stratos Idreos. 2025. TorchTitan: One-stop
                                                                                         PyTorch Native Solution for Production Ready LLM Pre-training.              https:
tions. AMD, the AMD Arrow logo, AMD CDNA™, AMD Instinct™,                                //doi.org/10.48550/arXiv.2410.06511 arXiv:2410.06511 [cs.CL]
AMD ROCm™, AMD Infinity Cache™, AMD Infinity Fabric™, and                           [12] LLVM Project. 2025. User Guide for the AMDGPU Backend: Memory Model.
                                                                                         https://llvm.org/docs/AMDGPUUsage.html#memory-model. [Online; accessed
combinations thereof are trademarks of Advanced Micro Devices,                           6-August-2025].
Inc. Other product names used in this publication are for identifi-                 [13] NVIDIA. 2025. NCCL: Optimized Primitives for Collective Multi-GPU Communi-
cation purposes only and may be trademarks of their respective                           cation. https://github.com/NVIDIA/nccl [Online; accessed 27-October-2025].
                                                                                    [14] NVIDIA Corporation. 2024. CUTLASS: CUDA Templates and Python DSLs for
companies.                                                                               High-Performance Linear Algebra. https://github.com/NVIDIA/cutlass. Version
                                                                                         4.3.0. [Online; accessed 2-November-2025].
                                                                                    [15] NVIDIA Corporation. 2025. CUDA C++ Programming Guide. https://docs.nvidia.
References                                                                               com/cuda/cuda-c-programming-guide/. [Online; accessed 7-August-2025].
 [1] Advanced Micro Devices, Inc. 2025. HIP Documentation (ROCm Software Future     [16] NVIDIA Corporation. 2025. NVSHMEM: OpenSHMEM-Based Parallel Program-
     Release). https://rocm.docs.amd.com/projects/HIP/en/docs-develop/index.html.        ming Interface for NVIDIA GPUs. https://github.com/NVIDIA/nvshmem. [On-
     [Online; accessed 6-August-2025].                                                   line; accessed 6-November-2025].
 [2] Advanced Micro Devices, Inc. 2025. Introducing AMD CDNA™ 3 Architecture.       [17] OpenSHMEM. 2012. OpenSHMEM Specification. http://openshmem.org/site/
     Technical Report. Advanced Micro Devices, Inc. [Online; accessed 6-August-          specification/. [Online; accessed 6-August-2025].
     2025].                                                                         [18] OpenXLA Project. 2025. XLA: Accelerated Linear Algebra Compiler for GPUs,
 [3] Advanced Micro Devices, Inc. 2025. ROCm OpenSHMEM (rocSHMEM). https:                CPUs, and ML Accelerators. https://github.com/openxla/xla. [Online; accessed
     //github.com/ROCm/rocSHMEM. [Online; accessed 6-August-2025].                       6-November-2025].
 [4] AMD. 2025. RCCL: ROCm Communication Collectives Library. https://github.       [19] Muhammad Osama, Robert Swann, Krishnan Sangaiah, Saurabh Singh, Ganesh
     com/ROCm/rccl [Online; accessed 27-October-2025].                                   Dasika, and Rakesh Bhardwaj. 2025. Deep Dive into the MI300 Compute
 [5] Muhammad Awad, Muhammad Osama, and Brandon Potter. 2025. Iris: First-Class          and Memory Partition Modes. https://rocm.blogs.amd.com/software-tools-
     Multi-GPU Programming Experience in Triton. https://doi.org/10.5281/zenodo.         optimization/compute-memory-modes/README.html. Accessed: 2025-08-10.
     17382307                                                                       [20] PyTorch Foundation. 2025. PyTorch 2.9 Release Blog. https://pytorch.org/blog/
 [6] Li-Wen Chang, Wenlei Bao, Qi Hou, Chengquan Jiang, Ningxin Zheng, Yinmin            pytorch-2.9-release/. [Online; accessed 6-November-2025].
     Zhong, Xuanrun Zhang, Zuquan Song, Chengji Yao, Ziheng Jiang, Haibin Lin,
Iris: First-Class Multi-GPU Programming Experience in Triton


[21] Ben Sander. 2025. Understanding Peak, Max-Achievable & Delivered FLOPs, Part
     1. https://rocm.blogs.amd.com/software-tools-optimization/Understanding_
     Peak_and_Max-Achievable_FLOPS/README.html.
[22] Ben Sander, Evan Masters, Babak Poursartip, and Henry Ho. 2025. Measuring
     Max-Achievable FLOPs – Part 2. https://rocm.blogs.amd.com/software-tools-
     optimization/measuring-max-achievable-flops-part2/README.html.
[23] Benjamin Spector, Jordan Juravsky, Stuart Sul, Owen Dugan, Dylan Lim, Dan
     Fu, Simran Arora, and Chris Ré. 2025. Look Ma, No Bubbles! Designing a Low-
     Latency Megakernel for Llama-1B. https://hazyresearch.stanford.edu/blog/2025-
     05-27-no-bubbles. [Online; accessed 10-August-2025].
[24] Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, and Christopher
     Ré. 2024. ThunderKittens: Simple, Fast, and Adorable AI Kernels.           https:
     //doi.org/10.48550/arXiv.2410.20399 arXiv:2410.20399 [cs.LG]
[25] Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, and Christo-
     pher Ré. 2024. ThunderKittens: Simple, Fast, and Adorable AI Kernels.
     arXiv:2410.20399 [cs.LG] https://arxiv.org/abs/2410.20399
[26] Philippe Tillet, H. T. Kung, and David Cox. 2019. Triton: An Intermediate Lan-
     guage and Compiler for Tiled Neural Network Computations. In Proceedings of
     the 3rd ACM SIGPLAN International Workshop on Machine Learning and Program-
     ming Languages (MAPL ’19) (MAPL 2019). 1–10. https://doi.org/10.1145/3315508.
     3329973
[27] Octavian Alexandru Trifan, Karthik Sangaiah, Muhammad Awad, Muhammad
     Osama, Sumanth Gudaparthi, Alexandru Nicolau, Alexander Veidenbaum, and
     Ganesh Dasika. 2025. Eliminating Multi-GPU Performance Taxes: A Systems
     Approach to Efficient Distributed LLMs. https://doi.org/10.48550/arXiv.2511.
     02168 arXiv:2511.02168 [cs.DC]
[28] Size Zheng, Wenlei Bao, Qi Hou, Xuegui Zheng, Jin Fang, Chenhui Huang,
     Tianqi Li, Haojie Duanmu, Renze Chen, Ruifan Xu, Yifan Guo, Ningxin Zheng,
     Ziheng Jiang, Xinyi Di, Dongyang Wang, Jianxi Ye, Haibin Lin, Li-Wen Chang,
     Liqiang Lu, Yun Liang, Jidong Zhai, and Xin Liu. 2025. Triton-Distributed:
     Programming Overlapping Kernels on Distributed AI Systems with the Triton
     Compiler. https://doi.org/10.48550/arXiv.2504.19442 arXiv:2504.19442 [cs.DC]
