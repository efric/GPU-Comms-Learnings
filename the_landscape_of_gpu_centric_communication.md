                                         The Landscape of GPU-Centric Communication

                                         DIDEM UNAT, Koç University, Turkey
                                         ILYAS TURIMBETOV, Koç University, Turkey
                                         MOHAMMED KEFAH TAHA ISSA, Koç University, Turkey
                                         DOĞAN SAĞBILI, Koç University, Turkey
                                         FLAVIO VELLA, University of Trento, Italy
                                         DANIELE DE SENSI, Sapienza University of Rome, Italy
                                         ISMAYIL ISMAYILOV, Koç University, Turkey
arXiv:2409.09874v4 [cs.DC] 23 Apr 2026




                                         In recent years, GPUs have become the preferred accelerators for HPC and ML applications due to their parallelism and high memory
                                         bandwidth. While GPUs boost computation, inter-GPU communication can create scalability bottlenecks, especially as the number of
                                         GPUs per node and cluster grows. Traditionally, the CPU managed multi-GPU communication, but advancements in GPU-centric
                                         communication now challenge this CPU dominance by reducing its involvement, granting GPUs more autonomy in communication
                                         tasks, and addressing mismatches in multi-GPU communication and computation.
                                            This paper provides a landscape of GPU-centric communication, focusing on vendor mechanisms and user-level library supports. It
                                         aims to clarify the complexities and diverse options in this field, define the terminology, and categorize existing approaches within
                                         and across nodes. The paper discusses vendor-provided mechanisms for communication and memory management in multi-GPU
                                         execution and reviews major communication libraries, their benefits, challenges, and performance insights. Then, it explores key
                                         research paradigms, future outlooks, and open research questions. By extensively describing GPU-centric communication techniques
                                         across the software and hardware stacks, we provide researchers, programmers, engineers, and library designers insights on how to
                                         exploit multi-GPU systems at their best.

                                         CCS Concepts: • Networks → Programming interfaces; • Computing methodologies → Parallel programming languages; •
                                         Hardware → Communication hardware, interfaces and storage; • Computer systems organization → Single instruction,
                                         multiple data.

                                         Additional Key Words and Phrases: GPUs, communication, MPI, NVSHMEM, NCCL, RCCL, Peer-to-peer Communication, Collective
                                         Communication, GPUDirect Technologies.

                                         ACM Reference Format:
                                         Didem Unat, Ilyas Turimbetov, Mohammed Kefah Taha Issa, Doğan Sağbili, Flavio Vella, Daniele De Sensi, and Ismayil Ismayilov. 2026.
                                         The Landscape of GPU-Centric Communication. ACM Comput. Surv. 1, 1, Article 1 (April 2026), 37 pages. https://doi.org/XXXXXXX.
                                         XXXXXXX

                                         Authors’ addresses: Didem Unat, Koç University, Istanbul, Turkey, dunat@ku.edu.tr; Ilyas Turimbetov, Koç University, Istanbul, Turkey, iturimbetov18@
                                         ku.edu.tr; Mohammed Kefah Taha Issa, Koç University, Istanbul, Turkey, MISSA18@ku.edu.tr; Doğan Sağbili, Koç University, Istanbul, Turkey, dsagbili17@
                                         ku.edu.tr; Flavio Vella, University of Trento, Trento, Italy, flavio.vella@unitn.it; Daniele De Sensi, Sapienza University of Rome, Rome, Italy, desensi@di.
                                         uniroma1.it; Ismayil Ismayilov, Koç University, Istanbul, Turkey, iismayilov21@ku.edu.tr.


                                         Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
                                         made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
                                         of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to
                                         redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
                                         © 2026 Association for Computing Machinery.
                                         Manuscript submitted to ACM


                                         Manuscript submitted to ACM                                                                                                                               1
2                                                                                                            D. Unat et al.


1    Introduction
In recent years, GPUs have become the accelerator of choice for a vast array of applications across both HPC and ML.
This quick adoption, driven mainly by the GPU’s massive parallelism and high memory bandwidth, means that most of
modern cloud and HPC capability is now concentrated in clusters of GPUs. As of November 2025, 9 of the 10 leading
Top500 supercomputers rely on GPU clusters for acceleration and this trend is likely to continue [79]. The only system,
Fugaku, without GPUs employs a highly vectorized CPU architecture combined with High-Bandwidth Memory.
    While using scores of GPUs has been shown to significantly accelerate computation, communication between the
GPUs can quickly become a scalability bottleneck [136] [149]. Traditionally, multi-GPU communication, both within
and across nodes, had always been the responsibility of the CPU. From their conception, GPUs were thought of as
devices that can supply large amounts of computation but that are also inherently dependent on the CPU for auxiliary
tasks like communication. In this CPU-centric model of execution, the routines relaying data for consumption by the
GPUs are oblivious to the GPUs’ existence.
    In the last decade, several advancements, broadly referred to as GPU-centric communication, have sought to challenge
the CPU’s hegemony on multi-GPU execution. At a high level, these advancements reduce the CPU’s involvement
in the critical path of execution, give the GPU more autonomy in initiating and synchronizing communication and
attempt to address the semantic mismatch between multi-GPU communication and computation.
    In this paper, we present a comprehensive review of GPU-centric communication with respect to vendor mechanism
and user-level library supports. Our goal with this survey is to help allay the general state of confusion that arises when
a prospective researcher begins wading into the field. We hope to help programmers, engineers, programming model
and library designers understand the complexity and diversity of available options because GPU-centric communication
spans a very broad spectrum of approaches, including hardware innovations like proprietary GPU-to-GPU interconnects
and software mechanisms. These mechanisms have distinct benefits and challenges making it unclear when and where
they should be preferred. The picture is made even murkier by inconsistent terminology used across literature and
vendor differences in offerings.
    We organize the paper as follows:

     • In Section 2, we define the terminology, provide a definition for GPU-centric communication. We taxonomize
        and abstract the existing approaches within a node and across nodes to eliminate any confusion.
     • In Section 3, we present the history and discuss the vendor-provided mechanisms for enabling communication
        and networking, managing memory across devices in a multi-GPU execution. These mechanisms are used as
        building blocks for higher-level GPU-centric software libraries.
     • In Section 4, we list and compare the main communication libraries for both intra- and inter-node setups, discuss
        their benefits and challenges and rely on existing benchmarking works to provide insights into their performance.
     • In Section 5, we discuss the main research paradigms underlying GPU-centric communication, provide an outlook
        on the field and present open research questions.

    Some methods and technologies discussed in this survey are inherently tied to proprietary ecosystems. While a
substantial portion of the available literature and deployed systems focuses on NVIDIA technologies, we present
a balanced perspective by incorporating corresponding mechanisms from other vendors, such as AMD and Intel,
whenever information is available. At the same time, several capabilities remain exclusive to specific vendors; we
explicitly identify these vendor-specific features to ensure clarity and completeness.
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                             3

            Type                 API      Data Path    Examples
             1 Host Native       Host     Host         cuda/hipMemcpy and other host-side comm. (No P2P)
             2 Host-Controlled   Host     Device       cuda/hipMemcpy and other host-side comm. (P2P)
                                                       NVSHMEM / ROCSHMEM / Intel SHMEM, GPU-side,
             3 Device Native     Device   Device
                                                       direct load/store (P2P)
             4 Host Fallback     Device   Host         Direct load/store (No P2P)
                                   Table 1. Types of intra-node communication methods




2     Terminology and Communication Types
We can loosely define GPU-centric communication as mechanisms that reduce the involvement of the CPU in the critical
path of multi-GPU execution. This is a very broad definition covering a wide spectrum of solutions, involving both the
vendor-level improvements that grant GPUs autonomy in communication and user-level implementations that leverage
those improvements. To make this distinction clear, we discuss them in separate sections. In Section 3, we focus on
communication mechanisms and primitives provided natively as part of the NVIDIA CUDA and AMD ROCm runtimes.
In Section 4, we discuss how these mechanisms give rise to higher-level, GPU-centric communication libraries and
present both vendor-provided software support by AMD, Intel and NVIDIA as well as other industry and academic
solutions.
    We also point out the distinction between communication within a node (intra-node) and communication across
nodes (inter-node). A single GPU-accelerated node comprises a single shared memory host with multiple GPU cards
attached. When communicating within a node, any given GPU can be controlled by a single thread or process, with a
shared memory and address space. A multi-node system has multiple such nodes where a different process controls
each GPU, and memory is not shared between processes running on different nodes. The communication landscape
changes depending on the setup used, since inter-node communication requires handling the GPU-NIC interaction and
across-process communication.

2.1    Intra-Node Communication
Despite classification of communication methods into GPU- and CPU-side is commonly used and can be sufficient
for an end-user, it is not always explanatory and accurate. To avoid the vagueness of definitions, we divide the
communication methods into several types. These types are based on the executor of each of the operations performed
during communication. We define two main operations needed for the communication to take place in the intra-node
scenario and four operations for the inter-node one. In the intra-node case, the two components of a communication
call are:
      • API. Defines where the communication API call is made by the programmer or library.
      • Data path. Indicates who participates in the data movement and shows the corresponding data path.
The examples showing the classification of intra-node communication mechanism, together with figures depicting the
data paths are shown on Table 1 and Figure 1.
    The communication method in 1 , referred as host native, is made on the host side and does not involve direct P2P
(peer-to-peer) access between the devices. These involve all methods that can be launched on the host side with P2P
access disabled. Otherwise, as 2 (host-controlled) shows, the communication does not involve an extra copy to the
host memory, and passes directly through PCIe, NVLink or Infinity Fabric interconnects. GPUCCL (i.e., NCCL, RCCL,
                                                                                              Manuscript submitted to ACM
4                                                                                                                      D. Unat et al.

             Host Native                  Host-​Controlled                Device Native                    Host Fallback

    1               API
                                     2              API
                                                                    3                               4
                 CPU                           CPU                                CPU                            CPU
                                                                                                    API

                                                                            API
                                                                                                        GPU                GPU
      GPU                 GPU            GPU              GPU         GPU               GPU
                                                                                                  Legend   Data path       API call API



            Fig. 1. Data paths and API calls of intra-node communication methods. Figure is available under CC-BY [150].



oneCCL, etc), GPU-aware MPI and *memcpy operations have host-side API, so they may belong to both 1 and 2 . By
enabling direct access to peer device memory, the device-side API eliminates CPU involvement from both the data and
control paths in intra-node communication, as illustrated by 3 , referred to as device native, in Figure 1. NVSHMEM,
ROCSHMEM, Intel SHMEM offer host-side API as well, but require P2P access, so their host-side APIs belong to 2 and
device-side API is of type 3 . In-kernel P2P direct load and stores offer similar functionality and belong to type 3 , but
work even with P2P access disabled, which is type 4 , where data path falls back to the host.


2.2      Inter-Node Communication
The inter-node scenarios are more diverse since interaction with the NIC has to take place. The details of each method’s
implementations may involve complicated data paths and decision making. We distinguish four main components of
inter-node communication for a simpler classification. Apart from the API and the data path (to the NIC) used in the
intra-node scenario, there are two additional components involving interaction with the NIC:

        • Register/construct messages. This step involves the construction of data packets and their registration on the
          NIC.
        • Trigger communication. It defines who rings the doorbell on the NIC to issue data transfer.

    We identify five main categories when classifying inter-node communication. From 1 to 5 , as Table 2 and Figure
2 show, more components of communication calls were moved to the GPU side, while the data transfer path saw a
reduction in the number of data copies required to reach the NIC. Over the years, the communication methods and the
corresponding technologies depict the optimizations of both data transfer and communication control, which will be
discussed in Section 3. First, entirely CPU-side communication method in 1 was available, which has been improved
by removal of extra copy between CPU-GPU and CPU-NIC buffers in 2 with the help of shared pinned memory
between GPU and NIC. This reduced the latency for GPU-NIC transfers. After that, GPU RDMA (Remote Direct Memory
Access) 3 facilitated direct access to the GPU memory from NIC over PCIe minimizing the data path between them.
4 represents GPU-triggered communication technologies such as GPUDirect Async and GPU-TN [73], where the GPU
has become capable to initiate communications, given that CPU prepared the packets on the NIC in advance. 5 moves
the packet preparation and interaction with the NIC entirely to the GPU as well, making device-native communication
possible.
    The types given in Table 2 and Figure 2 do not reflect all possible combinations, since some libraries, based on the
configuration and the available hardware may lead to different combinations of data paths and control. For example,
without the RDMA technology, even with GPU-side communication control, the data path will involve the host memory.
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                                                                            5

        Type                             API            Register Trigger             Data path                                 Examples
                                                                                                                               GPU-Aware MPI before
         1 Host Native                   Host           Host         Host            Through host (2 copies)
                                                                                                                                2
                                                                                                                               GPUDirect 1.0 in GPU-
         2 Pinned Host Native            Host           Host         Host            Through host (1 copy)                     Aware MPI, NCCL,
                                                                                                                               RCCL, oneCCL
                                                                                                                               GPUDirect      RDMA,
                                                                                                                               ROCmRDMA in GPU-
         3 GPU RDMA                      D/H            Host         Host            Direct                                    Aware MPI, NCCL,
                                                                                                                               NVSHMEM, ROCSH-
                                                                                                                               MEM
                                                                                                                               GPUDirect Async in
                                                                                                                               GPU-Aware MPI, NCCL
         4 GPU-Triggered                 D/H            Host         Device          Depends on 3
                                                                                                                               v2.28, NVSHMEM, ROC-
                                                                                                                               SHMEM
                                                                                                                               NVSHMEM with IBGDA,
         5 Device Native                 Device         Device       Device          Depends on 3
                                                                                                                               GPUrdma
Table 2. Types of inter-node communication methods. Cells in bold refer to where a change or optimization has been made. D/H
means that both device-side and host-side API calls may belong to this type.




             Host Native                   Pinned Host Native                  GPU RDMA                       GPU-​Triggered                  Device Native
    1                               2                                3                              4                                   5
                   NIC                            NIC                              NIC                               NIC                          NIC

              T                               T                          T                                                       T                              T
        CPU                GPU          CPU             GPU                  CPU          GPU             CPU              GPU              CPU           GPU
             API                           API                                                                                                                  API




    Legend           Data path          Message construction     T       Triggering communication   API   API call



                         Fig. 2. Inter-node communication data and control paths. Figure is available under CC-BY [150].




For instance, Intel SHMEM uses a proxy thread on the host to execute actual RDMA using a standard OpenSHMEM
library even though the communication call is issued within a device kernel running on the GPU.


3       Vendor Mechanisms
In this section we discuss the vendor-provided mechanisms for enabling communication and networking, managing
memory across devices in a multi-GPU execution. These mechanisms are provided by GPU programming model
runtimes or as part of the extended APIs. The technologies are classified into four categories: memory managers,
GPUDirect technologies, hardware, and libraries. Figure 3 summarizes the technologies provided by NVIDIA, detailing
their timeline and availability.
     Next, we introduce the memory management mechanisms and GPUDirect technologies, followed by the hardware
support that served as precursors and ultimately made these communication methods viable. These technologies form
the backbone of the higher-level GPU-centric libraries, which are discussed in Section 4.
                                                                                                                                     Manuscript submitted to ACM
6                                                                                                                                                                                               D. Unat et al.

CUDA 2.2                 3.0            4.0            4.1            5.0          6.0     6.5               8.0                      9.0     10.0             11.0               11.8   12.0                  13.0
    2006          2010           2011           2012                        2013          2014        2016                     2017                     2020           2022                     2024           2026




                                                                   Kepler




                                                                                                         Pascal




                                                                                                                                                                                                   Blackwell
                                                                                                                                               Turing




                                                                                                                                                                         Hopper
        Tesla




                                                                                           Maxwell
                     Fermi




                                                                                                                                  Volta




                                                                                                                                                           Ampere




                                                                                                                                                                                                                  Rubin
                                                                                                                                                                                     NVLink 4.0
                                                                                                                                                                                     NVSwitch

                               GPUDirect        UvA          IPC                         UvM                         NVLink 1.0             NVLink 2.0              NVLink 3.0           NVLink 5.0
                               1.0 (NIC)                                                                                                    NVSwitch                NVSwitch              NVSwitch
                                              P2P DMA                       GPUDirect RDMA                         GPUDirect Async                                                                NVLink 6.0
                 Pinned memory                GPUDirect 2.0 (P2P)                                                                                                                                  NVSwitch

                                                                            GDRCopy                                libgdsync                UCX

                                                             GPU-​aware MPI                                        NCCL                                 NVSHMEM


        Legend                      Memory managers                GPUDirect Tech                    Hardware             User-​level libraries                     Developer libraries



Fig. 3. Timeline of NVIDIA technologies enabling GPU-centric communication and networking. Figure is available under CC-BY
[150].



3.1        Memory Management Mechanisms
3.1.1 Page-Locked / Pinned Memory By default, memory allocated on the host using device malloc (e.g. cudaMalloc(),
hipMalloc() is pageable and is not accessible to the GPU. When a transfer between pageable host memory and device
memory is performed, the GPU runtime must first stage the host data through a temporary buffer in page-locked memory
and then copy the data from page-locked memory to the GPU. To avoid the pageable → page-locked memory copy,
cudaMallocHost() allows allocating page-locked memory directly, skipping the intermediate copy stage. Because of this,
page-locked memory is also referred to as zero-copy or pinned memory [52, 93].
     Pinned memory, known for its high bandwidth and low latencies in host-device transfers [115], efficiently coordinates
CPU-GPU execution by enabling direct system-wide access. It is also utilized with GPUDirect RDMA for improved inter-
node communication [75]. However, its physical memory locking can lead to high memory consumption, potentially
impacting system performance with excessive allocations [52].

3.1.2           Unified Virtual Addressing (UVA) UVA is a memory management technique which allows all GPUs and
CPUs within a node to share the same unified virtual address space [87, 93]. Prior to UVA, host ↔ device and device
↔ device copies had to explicitly specify the direction of transfer. With UVA, the physical memory location can be
inferred from pointer values, thus, reducing the overhead of managing separate memory spaces and enabling libraries
to simplify their interfaces [128].

3.1.3           IPC In early GPU runtimes versions, pointers could not be accessed across process boundaries, so memory
copies between GPU buffers had to go through the host, creating a bottleneck. To overcome this limitation, Inter-Process
Communication (IPC) enables processes on the same node to access device buffers of other processes without additional
copies [90]. With IPC, memory handles are created and passed between processes using standard IPC mechanisms,
resulting in lower latencies than staging copies through the host.

3.1.4           Unified Virtual Memory (UVM) UVM allows for the allocation of managed memory through cudaMalloc-
Managed() calls by creating a single address space accessible to all processors within a single node. UVM works by
dividing the requested memory into pages that are resident on the CPU. The programmer can access memory on a
device without explicit copies. If a memory access is part of a page that is not on the device, the UVM-driven triggers a
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                            7


page-fault that automatically migrates the page to the requesting device. The UVM driver can also evict pages from a
given device back to host memory when the total paged memory size exceeds device memory [93].
   UVM provides several benefits in regard to programmability. First, programmers are exposed to a single unified
address space that they can access as if the whole allocated chunk of memory is resident on a single GPU. Any copies
occurring around the system are implicit and hidden from the programmer’s view. Additionally, UVM allows memory
oversubscription whereby more memory can be allocated than all the multi-GPU device memory combined. This is
possible since most of the memory can stay on the CPU and be paged in whenever a given device requests it [92, 134].


3.2     GPUDirect Technologies
3.2.1 GPUDirect 1.0 (NIC) GPUDirect 1.0 allowed GPUs and NICs to share the same pinned memory region. Prior,
the pinned memory regions in system memory for GPUs and the NIC were separate. By implication, to communicate
GPU data across nodes, the GPU first copied the data to its pinned memory region, the CPU then copied it to the NIC’s
memory region, only then it can be accessed by the NIC, which sent it across the network, as shown in Figure 2 1 .
The intermediate CPU-initiated copy from GPU → NIC pinned memory regions adds CPU overhead and increases the
latency for GPU communication. GPUDirect 1.0 introduced a shared memory GPU-NIC pinned memory region, thus,
avoiding the intermediate CPU-initiated copy [125, 132].

3.2.2    GPUDirect 2.0 (Peer-to-Peer) Along with the introduction of UVA, the CUDA 4.0 release added support for
direct peer-to-peer communication among GPUs in a single node that share the same PCIe root complex [87]. This
functionality was encapsulated in a technology known as GPUDirect 2.0 or GPUDirect P2P. Instead of staging data
through the host, GPUs could now directly access each other’s memory over PCIe, establishing, for the first time, a
direct GPU-to-GPU data path. These changes led to two new communication mechanisms: P2P DMA Copies whereby
a cudaMemcpy call would trigger a DMA transfer directly between source and target GPU memories and P2P Direct
Load / Stores using which the GPUs could directly access data by dereferencing pointers to the remote GPU buffers.
GPUDirect P2P also added support for NVLink (Section 3.4) when the latter technology was introduced [88, 98, 125].
   GPUDirect P2P provided two main benefits. It eliminated redundant GPU ↔ CPU copies and host buffers, which
were required when the transfers were staged through the CPU. Also, by eliding the need to maintain communication
buffers on the host and providing a new communication mechanism (P2P Direct Load / Stores), GPUDirect P2P increased
the convenience of multi-GPU programming [88].
   We note that P2P DMA Copies can also work without UVA support. If UVA is not enabled, P2P DMA Copies can be
performed using the cudaMemcpyPeer() variants by explicitly specifying the target GPU. However, P2P Direct Load /
Stores will not work without UVA as directly accessing a remote GPU’s pointer presumes a unified address space [93].

3.2.3    GPUDirect RDMA With the introduction of GPUDirect RDMA in CUDA 5.0, direct communication between
NVIDIA GPUs across nodes became feasible. GPUDirect RDMA facilitates a direct communication channel between GPUs
and third-party devices through standard PCIe features. The technology exposes segments of GPU memory on the PCIe
memory resource, referred to as the Base Address Register (BAR) region. This enables NICs to directly read/write GPU
memory without routing through the host [96]. Analogously, AMD offers ROCm RDMA (previously called ROCnRDMA)
[10, 12]. GPUDirect RDMA provides several optimizations to the data path, namely by eliminating additional copies to
host memory, reducing inherent latencies stemming from GPU-NIC interaction, increasing bandwidth and reducing
CPU overhead.
                                                                                             Manuscript submitted to ACM
8                                                                                                          D. Unat et al.


3.2.4    GPUDirect Async While previous GPUDirect technologies focused on improving the data path, GPUDirect
Async optimizes the control path between the GPU and the NIC. Introduced in CUDA 8.0, it enables GPUs to initiate
and synchronize network transfers, thereby reducing the CPU’s involvement in the critical path. GPUDirect Async
works by having the CPU pre-register messages, which the GPU kernel can then trigger by ringing a doorbell on the
NIC. As a result, the GPU can continue executing while the communication is being triggered, rather than needing to
stop for the CPU to initiate the communication, as was previously necessary [3, 4].
    Although GPUDirect Async has led to improvements in efforts to move the control path away from the CPU, it still
does not completely transfer the control path to the GPU since communication is limited to kernel launch boundaries.
Essentially, the GPU can only initiate messages previously registered by the CPU. Further improvements to GPUDirect
Async are implemented as part of the IBGDA transport in the NVSHMEM library (Section 4.3.1).


3.3     GPUNetIO
GPUNetIO [1] is a technology solution proposed by NVIDIA as part of DOCA (Datacenter-On-a-Chip Architecture) [34].
DOCA is a full-stack software framework designed to facilitate the development of applications for NVIDIA BlueField
Data Processing Units (DPUs) [22]. On non-RDMA networks GPUNetIO allows the GPU to send, receive, and process
network packets [1, 2]. On RDMA networks (both RoCE and InfiniBand) [34], from DOCA v2.7, GPUNetIO allows the
GPU to execute RDMA send and receive not only on kernel boundaries, but at any point during the kernel execution. In
a nutshell, GPUNetIO allows the GPU to interact with the NIC without any CPU intervention.
    On RDMA networks, the GPU kernel can wait (in blocking or nonblocking mode) for the completion of RDMA receive
operations. On non-RDMA networks, GPUNetIO provides semaphores that can be used explicitly within the kernel
for synchronization with the NIC when sending and receiving packets. Semaphores can also be used to synchronize
GPU kernels with the CPU (in case the packet processing is split between the CPU and the GPU) or with other CUDA
kernels (if the packet processing is split across multiple kernels).


3.4     Modern GPU-centric Interconnects
GPU-centric interconnect technologies provide high-bandwidth, low-latency communication between multiple GPUs
within a node, a capability critical for high-performance workloads, particularly in AI training and HPC.
    NVLink is a proprietary interconnect technology for NVIDIA GPUs. Its design addresses the bandwidth limitations
of PCIe, which has been observed to be a transfer bottleneck in GPU-accelerated applications [75, 91]. Table 3 presents
the specifications of each generation of NVLink [45, 75, 99].


                                               Per   direction         Total    Aggregate
                               Number of                                                    Supported
                Generation                     Bandwidth per           Bi-Directional
                               NVLink slot                                                  Architecture
                                               NVLink (GB/sec)         Bandwidth (GB/sec)
                First          4               20                      160                  Pascal
                Second         6               25                      300                  Volta
                Third          12              25                      600                  Ampere
                Fourth         18              25                      900                  Hopper
                Fifth          18              50                      1800                 Blackwell
                                         Table 3. NVLink Generation Specifications


Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                                9


   Additionally, NVLink was also used to connect GPUs with the CPU for IBM Power8 and Power9 CPUs but with
the introduction to Grace Hopper Superchip, NVLink is used as a Chip-to-Chip (C2C) interconnect with 900 GB/sec
bi-directional bandwidth [75, 103]. Later with the introduction of Grace Blackwell Superchip, NVLink-C2C is used for
connecting Grace CPU with 2 Blackwell GPUs with a total of 3.6 TB/sec bidirectional bandwidth[102]. Fifth-generation
NVLink on NVIDIA Blackwell delivers 1.8TB/s bidirectional throughput per GPU, providing high-speed communication
among up to 576 GPUs.
   The introduction of NVLink optimized the bandwidth between NVIDIA GPUs, turning P2P communication into a
viable mechanism for intra-node communication and shifting the data path heavily in favor of GPUs. A disadvantage
of NVLink is that it is not self-routed meaning that if any two given GPUs do not have a direct NVLink connection
communication will have to be routed through an intermediate GPU [75]. This limitation is overcome by NVSwitch
[99], a backboard technology that can implement all-to-all connections between all GPUs. As an example, a DGX-2
node consists of 16 V100 GPUs that are all-to-all connected through NVLink and NVSwitch [95]. Starting from the third
generation, NVSwitch supports SHARP [104], which offloads allreduce operations to NVSwitch, allowing allreduce to
operate at full line rate [60].
   AMD’s alternative is Infinity Fabric/xGMI, used in modern AMD GPU accelerators (e.g., the AMD Instinct MI300X).
xGMI provides high-bandwidth communication between GPUs within a node. The architecture supports an aggregate
bidirectional inter-GPU bandwidth of 896 GB/s for an 8-GPU system. Unlike NVSwitch, AMD’s current interconnect
mesh is flat and does not rely on a switching fabric. This places some limits on scaling beyond 8 GPUs per node. Recently,
the Ultra Accelerator Link (UALink) consortium was established to develop a more open shared memory accelerator
interconnect, compatible with multiple technologies and vendors [81].
   Intel’s Xe-Link fabric is the high-bandwidth, fully connected intra-node interconnect used in systems such as Aurora
supercomputer, linking the Intel Data Center GPU Max (Ponte Vecchio) devices into a unified topology. In typical
Aurora nodes, six GPUs are connected in an all-to-all configuration, though Xe-Link can also support up to 8-way
arrangements where every GPU has direct links to all others. These links support load/store accesses, copy-engine
transfers, and remote atomic operations, enabling GPUs to directly access each other’s memory without host.

3.5   Discussion on Vendor Mechanisms
3.5.1 Impact of GPUDirect P2P and Direct Load/Store on Programming The introduction of GPUDirect P2P
marked a significant shift in the paradigm of multi-GPU execution, enabling direct communication between GPUs using
load and store operations from within the kernel. Direct Load/Store-based communication offers several benefits. First,
it allows the programmer to inline communication with computation. The programmer no longer has to rely on separate
models for communication and computation and can instead combine them within the GPU kernel [71, 117, 120]. Second,
Direct Load/Stores utilize the high levels of parallelism offered by the GPU and can achieve higher levels of bandwidth
and lower latencies compared to DMA copies [18, 114]. Third, Direct Load/Stores can implicitly overlap communication
with computation through the GPU’s inherent latency hiding capabilities. Given both the high levels of parallelism
granted by the GPU and the increasing bandwidth numbers offered by modern interconnects, the GPU has the capability
to hide latencies not only to local but remote memory as well [116, 117, 120]. This is another boon for the programmer
as the method of achieving overlap is shifted from a manual software-based approach implemented by the programmer
through streams and events to an automatic hardware-based overlap. Since the onus of communication/computation
overlap is passed from the programmer to the hardware, another implication is that the overlap will improve as the
hardware gets better at hiding memory latencies. Fourth, Direct Load / Stores expand the scope of applications that
                                                                                                 Manuscript submitted to ACM
10                                                                                                            D. Unat et al.


could be accelerated through multiple GPUs. Traditionally, applications with fine-grained communication patterns
achieved poor scalability on multi-GPU systems as computation frequently had to be interrupted and synchronized in
order for the CPU to initiate communication. With Direct Load / Stores from within the kernel, GPUs can adapt well
to fine-grained communication patterns. Finally, Direct Load / Stores allows communication to be triggered within
the kernel without leaving the GPU. This direction is particularly promising when combined with persistent kernels
[61, 165], where a single kernel is launched and maintains its execution across multiple work iterations by employing
an internal loop on the device, thereby minimizing kernel launch overhead. In fact, Spector et al. used direct Load/Store
on tensor-parallel inference with Llama-70B and implemented a megakernel to perform asynchronous communication,
overlapping them with compute and local memory operations [141].
     Despite the improvements conferred by Direct Load / Stores, there are several inherent challenges. First, a fundamental
challenge is that communication and computation contend for the same limited resource as they now both require large
volumes of GPU threads to make progress. This can be especially problematic when communication is implemented as a
separate kernel. If the computation kernel is launched first, it can potentially monopolize all GPU resources preventing
the communication kernel from being launched, effectively, eliminating any possibility of overlap. It is possible to
alleviate this issue by launching the communication stream with a higher priority so that it is always scheduled first.
We note that P2P DMA Copies do not have this issue as they use the GPU’s DMA / Copy Engines - a physically separate
resource - for communication [19]. Second, similar to single-GPU memory accesses, P2P Direct Load / Stores are
highly sensitive to memory coalescing with random non-coalesced access performing far worse than coalesced accesses
[18]. Such non-coalesced Direct Reads may expose remote memory latencies which are beyond the GPU scheduler’s
ability to hide, eventually, stalling execution. On a similar note, sporadic non-coalesced Direct Writes at sub-cacheline
granularities may dramatically underutilize the interconnect [83].

3.5.2 Limitation of GPUDirect RDMA A significant limitation of GPUDirect RDMA is that there are no guarantees
of consistency between GPU and NIC memories while a kernel is running. Consistency is guaranteed only by returning
control to the CPU by tearing down the kernel and launching a new kernel, thus, limiting communication to kernel
boundaries. This also implies that combining persistent kernels with GPU-initiated inter-node communication will
inevitably lead to data correctness issues [96]. Chu et al. get around this limitation by issuing a PCIe read from the NIC
to GPU memory which flushes the previous NIC writes to the GPU and guarantees memory ordering [31]. Since version
11.3, CUDA also offers the cudaDeviceFlushGPUDirectRDMAWrites() API which can be used to enforce consistency
similarly [38, 94]. While useful, CUDA still relies on the CPU to enforce GPU-NIC consistency. AMD, on the other hand,
has explicitly corrected the GPU-NIC consistency issues in the context of device-side communication from persistent
kernels and integrated the proposed fixes into ROC_SHMEM [51]. We further discuss this issue in the context of
CPU-free networking in Section 5.3.

3.5.3 Enabling Triggering Capability in GPU-Centric Communication In type 3 (GPUDirect/ROCn RDMA)
introduced in Section 2, the CPU is still responsible for the initial configuration of the system, the data transfer
preparation and initiating transfers. The first phase includes setting up network interfaces and loading GPU drivers. The
CPU registers GPU memory with the RDMA-capable NIC. This registration by host allows the NIC to directly access
GPU memory, bypassing the need for intermediary CPU steps during data transfer. In phase two, the CPU allocates
GPU memory buffers and ensures they are aligned correctly. These buffers will be used for efficient data transfer to
and from the GPU. Then, the CPU sets up GPU streams and events that manage and sequence the data transfers and,
guarantees the compilation of the work. Streams are used to queue operations, ensuring they are executed in the correct
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                                                     11



                                               Applications (ML, AI, HPC, Big Data)
                                         AI Frameworks                    HPC Libraries and Packages
                                     PyTorch     JAX    TensorFlow            PETSc      Trilinos      GROMACS


                                                 Communication APIs / Libraries
                                          MPI
                                    (Message Passing
                                                              NCCL      *CCL               SHMEM          IntelSHMEM

                                       Interface)             RCCL         oneCCL            NVSHMEM      rocSHMEM




                                    Communication Middleware / Transport                                   Vendor
                                               Abstractions                                               Software /
                                       UCX                     libfabric                                    SDKs

                                                                                                              DOCA
                                                 RDMA Software Stack
                                       RDMA-​core          libibverbs         provider libraries              OFED


                                                         Kernel / Device Drivers

                                                 Interconnects / Network Fabrics
                                      InfiniBand          RoCE            EFA          iWARP            Slingshot



Fig. 4. RDMA-oriented software stack for GPU-centric communication, showing the layers from applications and domain frameworks
to communication libraries, transport middleware, provider libraries, kernel drivers, network fabrics, and vendor software stacks.
Figure is available under CC-BY [150].



order. However, for truly low-latency applications, this cost may still represent a bottleneck as the mechanism relies on
several synchronization points through the streams [84, 85, 123].
   GPU-trigger communication in type 4 and 5 defined in Section 2 facilitates the offloading of both computation
and communication control paths to the GPU by removing the synchronization cost described above. Here, triggered
operations play a crucial role, as they are special tasks scheduled to execute only when specific conditions are met.
In the stream-triggered (ST) strategy, these operations manage data transfers and synchronization through the GPU
control processor, thereby reducing CPU involvement. Deferred Execution is another essential aspect, where the CPU
creates command descriptors with deferred execution semantics and appends them to the NIC command queue. These
descriptors are executed when the conditions specified by the GPU control operations are fulfilled. For example, in
the HPE Slingshot 11 NIC supports these deferred operations [84]1 , including sending and receiving communications
that are triggered when a hardware counter reaches a given threshold. This is possible by enabling specific command
queues (e.g., Libfabric Deferred Work Queues). The synchronization between the GPU control processor and the NIC
ensures the successful completion of communication operations through a special mechanism (e.g., in NVIDIA DOCA
this is implemented with GPUNetIO semaphores [1]).

1 2023. Libfabric Deferred Work Queue, https://ofiwg.github.io/libfabric/v1.9.1/man/fi_trigger.3.html

                                                                                                                       Manuscript submitted to ACM
12                                                                                                     D. Unat et al.


     To contextualize these vendor mechanisms within the broader communication ecosystem, Figure 4 summarizes the
software stack from applications and communication libraries down to middleware, drivers, and network fabrics. This
layered view helps connect the low-level mechanisms discussed in this section with the user-level communication
libraries reviewed next.

4     GPU-Centric Communication Libraries
We now discuss the main GPU-centric communication libraries namely, GPU-aware MPI, GPU-centric Collectives and
GPU-centric OpenSHMEM that have sprouted in recent years to ease multi-GPU programming.

4.1    GPU-Aware MPI
Given that MPI is the de facto lingua franca of HPC, much effort has gone into making it interoperable with GPU
programming models, culminating in GPU-Aware MPI implementations that can differentiate between host and device
buffers. Prior to GPU-Aware MPI, all multi-GPU communication had to be staged through the host incurring a device →
host copy on the source GPU and a host → device copy on the target GPU. Using a GPU-Aware MPI implementation,
on the other hand, a programmer can supply device buffers as parameters to the MPI call allowing communication to
use the direct GPU-to-GPU data path established by GPUDirect RDMA or ROCnRDMA. In the process, GPU-awareness
eliminates redundant host ↔ device copies and simplifies the communication code by eliding the need for host buffers.
     MVAPICH2 was the first MPI implementation to begin actively integrating GPU-awareness into its runtime. Early
MVAPICH2 work introduced basic GPU-awareness, transparently staging GPU-resident buffers through the host and
optimizing transfers via pipelining schemes for the host ↔ device and device ↔ device transfers. These pipelining
schemes were made possible by UVA, which allowed the library to differentiate between host and device pointers
without relying on user hints. The ensuing GPU-awareness led to performance improvements over the GPU-oblivious
version [156–158]. A follow-up work used CUDA IPC to optimize intra-node transfers, which prior had to be staged
through buffers in host memory [121]. Eventually, support was added for GPUDirect RDMA over the rendezvous
protocol allowing transfers to bypass the host and eliminate redundant host ↔ device copies. This reduced latencies;
however, bandwidth was limited due to existing architectural limitations [119]. A subsequent work added support
for GPUDirect RDMA over the eager protocol rectifying the bandwidth limitation and further reducing latencies.
Additionally, a new loopback mechanism and an early version of GDRCopy [97] were used to eliminate expensive
host ↔ device cudaMemcpys [135]. GDRCopy allows GPU memory to be mapped to the user address space which is
optimized for small message sizes with minimal overhead [38, 97, 129]. Another work extended point-to-point MPI calls
to support GPUDirect Async allowing the GPU to progress the communication enqueued by the CPU, thus, optimizing
the control path [152]. Other works have also increasingly focused on adding UVM-awareness to MVAPICH2-GDR
[16, 50, 77].
     Other major MPI implementations have also integrated GPU-awareness. Open MPI [111, 161] introduced CUDA-aware
support in version 1.7.0. When using its internal backend, computation-based collectives (e.g., MPI_Allreduce) may
still stage data through the host [40], a limitation avoided with UCX, which enables GPU-resident data movement and
reductions. Open MPI also leverages GDRCopy [97] for small messages and CUDA IPC for intra-node communication.
GPU-awareness extends to AMD devices through ROCm via UCX [9, 11, 112]. However, Khorassani et al. provide a
native ROCm-aware runtime for MVAPICH2 which outperforms Open MPI with UCX on a cluster of AMD GPUs [129].
Open MPI has integrated the Unified Collective Communication (UCC) framework [151], a component of the UCX
ecosystem designed to provide a unified interface for high-performance collective operations. UCC builds upon UCX’s
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                               13


transport layer and leverages its topology-aware mechanisms (such as awareness of NVLink, PCIe, and shared-memory
hierarchies) to select efficient collective algorithms for GPU buffers. This integration enables Open MPI to automatically
offload or accelerate collectives using the most suitable GPU interconnect path.
  Other mainstream implementations have followed a similar evolution. MPICH, for instance, introduced GPU support
in version 3.4 through its CH4 communication layer, which manages device buffers and selects efficient communication
paths. Support was later extended to AMD GPUs via ROCm (since version 4.0), with Intel GPU support under devel-
opment [48]. On Cray systems, users must enable GPU support explicitly by setting MPICH_GPU_SUPPORT_ENABLED=1
[55]. MPICH also integrates GDRCopy, GPUDirect RDMA, and IPC for optimized transfers. MPICH has adopted UCX
as a network module in its CH4 device layer, and ongoing work explores the integration of UCC to enable optimized
GPU collectives [33].


4.2   GPU-Centric Collectives (GPUCCL)
As deep learning models get ever larger, their compute requirements necessitate deploying training across multiple GPUs.
Given the prevalence of collective communication in deep learning training, NVIDIA, AMD, and Intel all provide highly
efficient collective communication libraries that are optimized for their respective GPU architectures. For simplicity
and clarity, we refer to these vendor-specific solutions as GPU Collective Communication Libraries (GPUCCL). They
have been integrated as a communication backend for several state-of-the-art deep learning frameworks, including
Pytorch, Tensorflow, MXNet, Caffe, CNTK, and Horovod [160].
  The implementation of GPU-aware collectives involving computation (like reduce/allreduce) in MPI were imple-
mented using GPU kernels for local reductions and CPU-initiated copies among the GPUs to perform aggregation.
This approach incurs several kernel launch and communication call latencies and, additionally, requires intermediate
buffers on the host. GPUCCL takes a different approach by implementing the communication and computation for the
collective together in a single kernel [89]. Next, we highlight the differences among vendor solutions with respect to
the features they support.


4.2.1 Comparison of Collective Communication Libraries by Vendors Table 4 lists the main features of the
collective communication libraries supported by three main vendors: NVIDIA’s NCCL, AMD’s RCCL and Intel’s OneCCL.
  Beyond its GPU-centric execution model, NCCL distinguishes itself through a specific philosophy of collective
algorithm design. NCCL adopts a uniform design principle in which all collectives, AllReduce, AllGather, Broadcast,
ReduceScatter, and AllToAll, are implemented as sequences of a small number of reusable communication primitives
(send, recv, reduce, copy) applied over logical topologies determined at communicator creation. Hu et al. [56] show that
NCCL maps each collective onto either a ring or tree communication graph, chosen for bandwidth- or latency-dominant
scenarios, and pipelines these operations using fixed-size chunks to maximize overlap between computation and
communication. A key architectural element of NCCL’s algorithm design is the use of parallel communication channels.
Each channel corresponds to an independent instance of the collective algorithm, operating over a slice of the message.
Channels allow NCCL to run multiple ring (or tree) stages in parallel, leveraging GPU thread-block parallelism and
enabling bandwidth saturation across NVLink or NIC paths. As observed by Hu et al., channel count directly influences
algorithmic throughput: too few channels reduce link utilization, whereas too many channels increase queueing
and synchronization overheads. Channels therefore act as a structural mechanism for decomposing a collective into
parallelizable sub-algorithms.
                                                                                                 Manuscript submitted to ACM
14                                                                                                               D. Unat et al.

Table 4. Comparison of vendor collective communication libraries in terms of collective primitives, execution model, accelerator
integration, and transport semantics. Abbreviations: B=broadcast, R=reduce, AR=allreduce, RS=reducescatter, AG/AGv=allgather and
variants, AA=all-to-all, P2P=point-to-point.


  Library                     NCCL                           RCCL                             oneCCL

  Collective primitives       AR, R, B, AG/AGv, RS, AA, AR, R, B, AG/AGv, RS, AA, AR, R, B, AG/AGv, RS, AA/AAv,
                              P2P                       P2P                       P2P (async)
  API model                   CUDA streams; GPU-kernel- HIP streams; GPU-kernel-driven Unified SYCL/L0 API across
                              driven                                                   CPU+GPU ranks
  Execution engine            GPU     kernels       (device- GPU kernels (tile-aware)         CPU workers + device queues
                              autonomous)
  Accelerator integration NVIDIA GPUs only; CPU proxy AMD MI-series multi-tile GPUs; CPUs + Intel PVC GPUs; unified
                          threads                     explicit locality              collective namespace
  Intra-node fabric           NVLink / NVSwitch              xGMI / Infinity Fabric           Xe-Link + shared memory
  Transport backend           NCCL-NET (IB/RoCE, TCP); NCCL-NET plugins; HIP-aware ATL with MPI or libfabric;
                              GPU-triggered NIC operations routing                 SYCL/L0 transport
  Progress & initiation       GPU-initiated NIC operations   GPU-initiated operations; tile- CPU-initiated network opera-
                                                             level routing                   tions; GPU via SYCL/L0
  Topology awareness          Automatic NVLink/NVSwitch XCD topology;            hop-aware Provided by MPI/libfabric
                              graph construction        routing
  Distinctive property        Fully GPU-autonomous commu- GPU-driven,       topology-aware Heterogeneous,          transport-
                              nication                    collectives                      modular model



     While NVIDIA provides NCCL, AMD offers an analogous library named RCCL (ROCm Collective Communication
Library) [14] that mirrors the NCCL API and programming abstractions, enabling compatibility for deep learning frame-
works on ROCm platforms. Although the APIs are nearly identical, the performance characteristics differ significantly
due to AMD’s unique hardware design. RCCL operates across complex multi-die topologies, such as MI250MI300 accel-
erators, where communication traverses multiple XCD or Infinity Fabric (IF) tiers with distinct bandwidth properties.
Understanding these heterogeneous bandwidth hierarchies is essential for efficient collective algorithm construction
and motivates a stronger emphasis on routing and ring-selection heuristics in RCCL [127].
     Beyond its GPU-centric execution model, RCCL develops a collective-algorithm design philosophy shaped by this
hierarchical topology. Like NCCL, RCCL implements its core collectives by using a small set of reusable primitives.
However, these primitives must be composed over a heterogeneous, multi-hop substrate, where algorithmic stages
frequently cross XCD boundaries by adding non-uniform hop. As a result, RCCL constructs logical rings and trees
that explicitly reflect tile-level connectivity and link asymmetries; ring segments may be lengthened, partitioned, or
reordered to minimize traversal of low-bandwidth IF paths. This design makes collective execution more tightly bound
to the physical GPU structure than on NVSwitch-based systems.
     RCCL also adopts NCCL-style parallel communication channels to pipeline collective execution, but channel strategy
is influenced by multi-tile GPU layout and ROCm partition modes (e.g., CPX/NPS4). Each channel processes a slice of
the message while attempting to localize work within individual XCDs to reduce cross-tile traffic, creating a form of
algorithm-level tile-aware parallelism. Hidayetoglu et al.[53] further observe that RCCL’s latency and scaling behavior
reflect both chunking strategy and the cost of multi-hop IF traversal, making channel configuration more sensitive to
placement than in homogeneous NVIDIA designs.
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                            15


   Despite these architectural and algorithmic challenges, RCCL implements the same high-level collective patterns
as NCCL—rings, trees, and hierarchical algorithms—and reuses the NCCL network-plugin ABI to support InfiniBand
and RoCE transports. This feature compatibility enables shared communication backends across vendors when the
underlying transport permits it. However, unlike NVIDIA-based systems equipped with NVSwitch fabrics, current
AMD systems lack a switch-based all-to-all interconnect, making topology-aware collective construction and tile-level
algorithm design particularly important for scaling RCCL on large multi-GPU nodes.
   In addition to NCCL and RCCL, Intel provides oneCCL (oneAPI Collective Communications Library) as part of the
oneAPI ecosystem for CPU and GPU clusters. In contrast to NCCL/RCCL—which execute collective operations inside
GPU kernels—oneCCL adopts a middleware-oriented design built over MPI and libfabric, delegating transport selection
and topology awareness to the underlying communication stack. Rather than providing a purely GPU-resident collective
engine, oneCCL exposes a portable set of collective communication primitives and APIs designed to operate uniformly
across CPUs, integrated GPUs, and discrete Xe-series accelerators. The implementation of collectives reflects a layered
orchestration model. GPU memory is managed through SYCL/Level Zero device queues, but the orchestration of
collective operations is executed primarily by CPU worker threads, which schedule and advance operations on behalf of
GPUs. The ATL (Abstract Transport Layer, a modular backend in oneCCL architecture, maps collective primitives onto
lower-level communication fabrics such as MPI or libfabric (OFI). The ATL determines transport semantics, message
progression, and endpoint selection, while Level Zero or SYCL facilitate intra-device and device–host data movement.
This architecture enables oneCCL to integrate multiple accelerator types into a common collective namespace. CPU
ranks and GPU ranks can participate in the same collective invocation, with oneCCL coordinating data movement
across host memory, PCIe, and Xe-Link fabrics. The API ensures that collective primitives operate independently of
whether the underlying buffer is allocated into CPU RAM, GPU HBM, or shared unified memory. This is particularly
important on systems like Aurora, where CPUs and discrete PVC GPUs form tightly coupled multi-accelerator nodes.
   Empirical analyses by Kwack et al. [68] show that oneCCL unifies collective communication across these diverse
endpoints, with transport-level behaviors delegated to MPI/libfabric on Slingshot networks. However, oneCCL’s
abstraction comes with implications for communication progress and GPU-centric collective design. Unlike NCCL and
RCCL, whose collective kernels run entirely on GPUs, oneCCL’s GPU participation remains dependent and host-driven.
Therefore, collective primitives operate through command queues and event dependencies, but Level Zero kernels do
not independently initiate network activity. As a result, overlap and latency characteristics depend on CPU worker
scheduling and the performance of the ATL backend.
   This distinction is highlighted by recent GPU-aware MPI studies [24, 25], which demonstrate that direct Level Zero
and IPC-based approaches can outperform oneCCL for GPU–GPU collectives when host intervention becomes a limiting
factor. Performance analysis study [53] further shows that oneCCL exhibits different scaling patterns than NCCL/RCCL
on hierarchical multi-GPU systems precisely because its collective primitives are executed through a hybrid CPU/GPU
control structure.

4.2.2   Further Work on Collectives Despite its significant benefits, GPUCCL presents several inherent challenges,
primarily stemming from resource contention and its static abstraction model. First, a fundamental issue of using GPU
threads for communication and computation is that the two routines now contend for the same limited resource. In this
case, if computation is scheduled ahead of the collective, it can monopolize all GPU resources, effectively serializing
the collective behind the computation. One workaround around this is to launch the collective on a higher-priority
stream such that it is always scheduled first [19, 89]. Moreover, GPUCCL’s execution model is tailored for regular
                                                                                              Manuscript submitted to ACM
16                                                                                                         D. Unat et al.


collective patterns where communication size and data type are statically expressed as host arguments. While suitable for
deep learning, this design lacks the flexibility needed for more dynamic communication scenarios [137]. Furthermore,
NCCL’s design often restricts interconnects to a single transfer mode (such as thread-copy over DMA-copy), and
its conservative synchronization methods hinder the implementation of advanced optimization techniques, such as
fine-grained double-buffering to effectively hide communication latency [131].
     To address these aforementioned issues, several collective communication libraries have been proposed; notably,
many of the solutions developed by industry partners are now open-sourced and readily available for use.
     Blink is a collective communication library with the express goal of achieving optimal link utilization. To do this,
Blink detects the underlying topology, models the topology as a graph, and then uses a technique known as packing
spanning trees to dynamically generate communication primitives. As a result, Blink was shown to reduce model
training time on an image classification task by 40% compared to NCCL2 [155].
     Dryden et al. present Aluminum, a GPU-aware library for large-scale training of deep neural networks. Aluminum ex-
tends NCCL with tree-based algorithms to avoid the latency bottlenecks of its default ring implementation. Additionally,
it adds support for non-blocking NCCL allreduce operations. For MPI, Aluminum gets around forced synchronizations
stemming from the MPI-GPU semantic mismatch by associating a single GPU stream with an MPI communicator
and synchronizing with respect only to that stream. The resulting optimizations bring about speedups compared to
GPU-aware MPI and NCCL-based implementations [43].
     Several recent research systems aim to extend or specialize the capabilities of vendor-provided GPUCCL libraries.
Among these, NCCLX [137], developed by Meta, introduces a transparent acceleration layer for NCCL specifically
targeting deep-learning workloads. NCCLX observes that most data-parallel training jobs repeatedly invoke the same
collective patterns which are predominantly AllReduce and AllGather—over fixed message shapes. By exploiting this
regularity, NCCLX constructs optimized execution paths inside NCCL’s existing kernels through collective fusion,
path coalescing, and specialized kernel pipelines that eliminate redundant synchronization and memory movement
stages. Crucially, NCCLX preserves NCCL’s public API and requires no modification to frameworks such as PyTorch or
TensorFlow, making it a performance enhancement for standard deep-learning training stack.
     Unified Collective Communication (UCC) [151] is an open-source project providing an API and library implementation
of high-performance and scalable collective operations, leveraging topology-aware algorithms and techniques such as
in-network computing and DPU offloading. It relies on UCX point-to-point communication, as well as on NCCL/RCCL,
SHARP, and others. UCC offers a portable, backend-agnostic framework that unifies collectives across CPUs, GPUs, and
DPUs by dynamically selecting among UCX-based implementations, SHARP offload, and vendor libraries such as NCCL
and RCCL. Similarly, Microsoft Collective Communication Library (MSCCL) [131], introduces a low-level GPU-centric
communication substrate built around device-resident primitives (put, signal, wait, flush) and a high-level DSL for
algorithm synthesis, enabling fine-grained, fused communication–computation kernels that outperform traditional
NCCL paths on emerging AI workloads such as LLMs. MSCCL++ outperforms NCCL and MSCCL by up to 2.8x and 1.6x
for small messages, and by up to 2.4x and 2.0x for large messages, respectively. Hierarchical Collective Communication
Library (HiCCL) [54] adopts a hierarchy-aware design that decomposes collective operations into universal primitives
(multicast, reduction, fence) and maps them onto multi-tier accelerator topologies—including GPU tiles, multi-GPU
nodes, and multi-NIC clusters—achieving portable performance across NVIDIA, AMD, and Intel systems. Many other
efforts are going in the direction of unifying the offloading collective operations into the MPI runtime [26].



Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                               17


  Finally, MPI-xCCL [26] integrates NCCL, RCCL, and HiCCL to create a hybrid execution model in which MPI
transparently offloads collective operations to vendor libraries when beneficial, while retaining MPI semantics and
enabling collectives not supported natively by GPUCCLs.
  Beyond portability and topology-awareness, recent research has also begun to explore collective communication
optimization along new axes such as energy efficiency and configuration auto-tuning. PCCL [64] introduces a power-
aware collective communication layer built on top of NCCL that exploits the observation that many collective kernels,
particularly AllReduce and AllGather operations in LLM workloads, are frequency-insensitive and can run at substantially
reduced GPU clock frequencies without measurable bandwidth loss. By integrating fine-grained DVFS management
into the collective call path, PCCL identifies the minimum safe GPU frequency for each collective, and automatically
inserts frequency-set and frequency-reset events around NCCL kernels.
  Complementary to power-centric optimization, many libries address the challenge of optimising a large configuration
space (algorithm, protocol, transport, channels, threads, chunk sizes) that the default GPUCCL cost model can be
sub-optimal for many message sizes, GPU topologies, and communication patterns [76, 163]. AutoCCL [163] provides an
automated, online tuning framework that profiles collective execution during the early iterations of training, classifies
parameters into implementation-level versus resource-allocation parameters. Unlike prior offline tuners, AutoCCL
directly accounts for communication–computation balancing, frequently achieving 1.2–1.8x bandwidth improvements
over NCCL on both PCIe and NVLink systems and up to 32% end-to-end iteration-time improvements for LLM workloads.
The approach highlights the growing need for collective runtimes that dynamically adapt their execution strategy to
hardware conditions, workload patterns, and runtime interference.

4.3   GPU-centric OpenSHMEM (GPUSHMEM)
GPU-centric OpenSHMEM runtimes are NVIDIA’s NVSHMEM[100], AMD’s ROC_SHMEM[13] and Intel SHMEM[20]
libraries. Given its earlier inception, we first discuss NVSHMEM and, in doing so, introduce concepts fundamental to all
three libraries. Note that NVSHMEM and ROC_SHMEM are GPU-centric PGAS runtimes designed from scratch for
their respective GPU ecosystems. Both libraries follow the OpenSHMEM programming model, but they introduce their
own runtime structures and calling conventions, lack the context abstraction present in the OpenSHMEM specification,
and expose host APIs that operate only on device-resident symmetric memory. In contrast, Intel SHMEM adheres
closely to the OpenSHMEM specification and supports both host and device pointers via a unified SYCL-based API.

4.3.1 NVSHMEM NVSHMEM is NVIDIA’s implementation of the OpenSHMEM specification for CUDA devices.
NVSHMEM is a Partitioned Global Address Space (PGAS) library that provides efficient one-sided put / get APIs for
processes to access remote data objects. NVSHMEM supports point-to-point and collective communication between
GPUs both within and across nodes [100].
  NVSHMEM works on the concept of a symmetric heap. During NVSHMEM initialization, each process that is mapped
to a GPU, referred to as a processing element (PE) reserves a block of GPU memory using nvshmem_malloc(). In
NVSHMEM, all memory allocations must be performed collectively, meaning that all symmetric memory regions within
the heap must have identical sizes and must be allocated at the same time. To access remote memory on a different PE,
a given PE requires the offset for the symmetric memory as well as the rank of the remote PE.
  In addition, NVSHMEM provides APIs for synchronizing a group of PEs. These APIs comprise signal-wait mechanisms
that can serve as a means for point-to-point synchronization and collective synchronization calls that can function as
global barriers. This feature is particularly important since there is a general lack of kernel-side global barriers, with
                                                                                                 Manuscript submitted to ACM
18                                                                                                         D. Unat et al.


the CPU typically performing the role of global synchronizer for devices. The capacity for a device to synchronize
efficiently across the device without terminating kernel execution is a crucial prerequisite for transferring the control
plane to the GPU.
     A notable attribute of NVSHMEM is that it offers both host-side and device-side APIs. The host-side APIs expose an
optional stream argument that can be used to implement communication-computation overlap. For certain calls, the
GPU-side variants provide the calls in three granularities: thread, thread block, and warp. The thread variant means
that the call should be performed by a single device thread and will be executed by that thread. The thread block
and warp variants use multiple threads to execute the communication call cooperatively. These variants should be
called by all threads in the corresponding thread block or warp. A previous performance comparison between the
host-side and device-side APIs found negligible differences in performance between the two, with host-side APIs slightly
outperforming the device-side variants [47]. This study was conducted using an early version of NVSHMEM (0.3.0),
which has since seen improvements in GPU-side API performance.
     As of version 2.7.0, NVSHMEM introduced the Infiniband GPUDirect Async (IBGDA) transport built on top of
GPUDirect Async [101]. The IBGDA transport allows GPUs to issue inter-node communication directly to the NIC,
bypassing the CPU entirely. Without IBGDA, device-side inter-node communication calls are performed through a
proxy thread on the CPU that triggers the corresponding NIC operations. This proxy thread consumes CPU resources
and creates a bottleneck in achieving peak NIC throughput for fine-grained transfers [113]. NVSHMEM with IBGDA
support, combined with persistent kernels, enables the complete transfer of both data and control paths to the GPU
and marks a significant shift towards fully autonomous multi-GPU execution. However, as discussed in Section 3.2.3,
GPUDirect RDMA only enforces GPU-NIC memory consistency across kernel boundaries. This inherent reliance on the
CPU for memory consistency is a potential obstacle toward truly autonomous multi-GPU execution. One workaround
is using a callback mechanism whereby the persistent kernel signals the CPU to perform a consistency-enforcing
API call (i.e., cudaDeviceFlushGPUDirectRDMAWrites()). The efficacy of this solution is unclear and warrants further
investigation. Enforcing GPU-NIC memory from inside the kernel is supported by ROC_SHMEM, which we discuss in
the next section.
     In recent years, NVSHMEM has been integrated as a communication backend into multiple runtimes. PETSc
implemented PetscSF, a scalable communication layer based on NVSHMEM, to complement their MPI-based approach,
which did not work well with CUDA stream semantics and prevented kernel launch pipelining [164]. Kokkos Remote
Spaces, adding distributed memory support to the Kokkos programming model, uses NVSHMEM as a communication
backend [32, 146]. An NVSHMEM implementation of the Kokkos Conjugate Gradient Solver outperforms the CUDA-
aware MPI implementation while significantly reducing the code base size [78]. Choi et al. use persistent kernels and
NVSHMEM to implement CharminG, a fully GPU-resident runtime system inspired by Charm++ [30]. The Livermore
Big Artificial Neural Network (LBANN) implements a spatial-parallel convolution using NVSHMEM that outperforms
MPI and Aluminium implementations [78]. QUDA, a library for lattice QCD computations, has used NVSHMEM and
persistent kernels for improved strong scaling of Dirac operators [72, 154].
     NVSHMEM has also been used outside of runtime-based approaches to achieve performance improvements. Chu et
al. combine NVSHMEM with persistent kernels to implement a state-of-the-art GPU-based key-value store [31]. Xie
et al. use NVSHMEM to implement a single-node multi-GPU sparse triangular solver (SpTRSV) that achieves good
performance scalability compared to a UVM-based design [162]. Ding et al. combine persistent kernels with NVSHMEM
for impressive performance in a sparse triangular solver (SpTRSV) on single- and multi-node setups. Atos implements
both persistent and discrete kernels with NVSHMEM-based communication to achieve state-of-the-art performance on
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                             19


multi-GPU BFS within and across nodes [27]. Wang et al. propose MGG, a system design that accelerates Graph Neural
Networks (GNNs) on multi-GPU systems using a GPU-centric software communication-computation pipeline with
NVSHMEM for fine-grained communication [159]. Ismayilov et al. use persistent kernels and device-side NVSHMEM to
implement fully GPU-side Jacobi 2D/3D and CG solvers that outperform CPU-controlled baselines. They reserve some
thread blocks for communication while others handle computation, a technique they call thread block specialization, to
achieve explicit device-side communication-computation overlap [61]. Punniyamurthy et al. use ROC_SHMEM and
persistent kernels to overlap embedding operations with collective communication in deep learning recommendation
models [122].

4.3.2   ROC_SHMEM ROC_SHMEM is AMD’s implementation of the OpenSHMEM specification for AMD GPUs.
ROC_SHMEM offers two communication backends. The first, known as GPU-IB, implements Infiniband on the GPU,
similar to NVSHMEM’s IBGDA transport. The second, called Reverse Offload (RO), uses host-side proxy threads and
offloads communication to the CPU. GPU-IB is the default backend and offers the best performance [13]. ROC_SHMEM
works almost identically to NVSHMEM and offers analogous APIs. However, there are several significant differences.
First, as mentioned in the previous section, NVSHMEM runs into GPU-NIC memory consistency problems when
intra-node communication is issued from persistent kernels. ROC_SHMEM, on the other hand, explicitly addresses
this issue and guarantees correctness when persistent kernels are being used. Hamidouche et al. analyze the GPU-NIC
memory mismatch stemming from the GPU’s relaxed memory model and propose changes integrated into ROC_SHMEM
[51]. This means that ROC_SHMEM provides completely CPU-free communication mechanism that can move the
entire flow of multi-GPU execution to the device.
   Second, ROC_SHMEM uses GPU shared memory (local data store (LDS) in AMD parlance) to store network state for
faster access. To the best of our knowledge, this optimization is not implemented in NVSHMEM. While this is most
likely beneficial for execution time, the increased shared memory message could limit occupancy and negatively impact
performance [7, 122].
   Third, prior versions of ROC_SHMEM required allocating symmetric buffers as uncacheable in order to prevent
stale data from being communicated. However, as AMD recently introduced intra-kernel cache flush instructions, the
data can be flushed before initiating the network transaction, allowing the data to be cached. No such instructions are
provided by NVIDIA, meaning that NVSHMEM buffers are likely allocated as uncacheable [122].

4.3.3   Intel SHMEM Intel recently introduced Intel SHMEM [20], the first GPU-aware implementation of OpenSH-
MEM for Intel GPUs. It allows SHMEM routines to operate directly on GPU memory and supports GPU-initiated
communication by embedding SHMEM calls inside SYCL kernels. Intel SHMEM integrates with the SYCL programming
model, offering a portable C++ interface for heterogeneous systems. In contrast, existing solutions such as NVSHMEM
and ROC_SHMEM provide similar capabilities but are tied to vendor-specific ecosystems.
   To maintain compatibility with the OpenSHMEM 1.5 specification, Intel SHMEM supports both device- and host-side
APIs for point-to-point operations, collective operations via the teams API, and SYCL-specific extensions for work-group
and sub-group communication. Inter-node communication is handled through a host proxy thread, which forwards
GPU-initiated operations to a standard OpenSHMEM backend. In other words, GPU issues SHMEM operations inside a
SYCL kernel. The GPU writes work queue entries into a memory region accessible by both GPU and host. The host
thread running on CPU detects the GPU-initiated RMA operation. The host executes actual RDMA using a standard
OpenSHMEM library. The current implementation relies on Sandia OpenSHMEM (SOS), leveraging its robust support
for OFI/libfabric transports and its capability to maintain a symmetric heap in GPU memory. This layered design enables
                                                                                               Manuscript submitted to ACM
20                                                                                                          D. Unat et al.


Intel SHMEM to provide functional equivalence with NVSHMEM and ROC_SHMEM while integrating cleanly with
Intel’s broader oneAPI and SYCL ecosystem.
     For intra-node communication, Intel SHMEM can perform GPU-to-GPU data movement directly without involving
the host CPU. This is achieved by allowing the GPU to issue SHMEM operations from within SYCL kernels, using
Intel’s Level Zero–based backend to translate these operations into: (1) direct GPU load/store transfers, when GPUs
share a unified memory fabric, or (2) GPU copy-engine transfers, which bypass the host CPU and use dedicated DMA
engines on the GPU.


4.4    Comparison and Discussion of User-Level Libraries
While GPU-aware MPI, GPUCCL, and GPUSHMEM all offer mechanisms for programming multi-GPU systems,
significant differences exist in their semantics and performance characteristics. Key distinctions include stream support,
API location, programming and performance.


4.4.1 Streaming Support GPUs operate on the concept of streams which are command queues that guarantee ordering
among GPU operations. The GPU scheduler ensures that kernels and other operations launched on a stream execute in
the order they were enqueued and do so with correct data dependencies. Since kernel launches are asynchronous and do
not block the host, GPU runtimes can pipeline kernel launches and overlap the launch latencies behind kernel execution.
The semantic mismatch between the MPI and GPU models is that MPI has no awareness of GPU streams. As a result, it is
not possible to enqueue an MPI call on a given GPU stream or for a GPU stream to wait on the completion of a pending
MPI routine. By implication, interlacing MPI calls with GPU kernels will require host-blocking synchronizations in
order to maintain data correctness. For example, before initiating an MPI send, the programmer has to block the
host to synchronize all streams which operate on the send buffer. Similarly, waiting on completion of pending MPI
communication will also require host-blocking synchronization. In either case, these forced synchronizations impair
kernel launch pipelining, prevent opportunities for overlap and force the programmer into alternating bulk phases of
communication and computation [43, 126, 147, 164].
     We see two possible non-mutually exclusive paths for resolving the semantic mismatch. The first is making MPI
runtimes stream-aware by adding an explicit stream parameter to MPI routines. This would solve the issue of impaired
kernel launch pipelining and allow MPI calls to seamlessly integrate into GPU runtimes. The second is providing the
option of device-initiated MPI calls. This would reduce the programmer burden of juggling two distinct programming
models and, additionally, provide implicit communication-computation overlap. Both directions have been explored in
the literature on a limited scale. The FLAT compiler automatically converts device-side MPI calls to their host-side
equivalents [80]. dCUDA implements device-side operations with MPI semantics but uses CPU helper threads for
the actual communication. They rely on the GPU’s inherent memory latency hiding capabilities to implicitly overlap
communication with computation, ultimately outperforming a GPU-aware MPI baseline [49]. Namashivayam et al.
explore new communication schemes to introduce GPU stream-awareness in MPI. They use the triggered operations
feature on HPE Slingshot 11 interconnect, allowing the CPU to enqueue communication and synchronization operations
to the NIC, which the GPU can then trigger. This reduces CPU involvement in the critical path and eliminates expensive
synchronizations. While inter-node experiments show some performance benefits, the proposed scheme falters in
intra-node setups as progress threads need to be used to emulate deferred execution semantics [85]. A follow-up work
eliminates progress threads for intra-node communication, opting to use P2P Direct Load Store-based GPU kernels and
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                                        21


GPU IPC-based mechanisms instead. The evaluation shows performance improvements over stream-oblivious MPI
baselines [84].
   More recently, MPIX streams [166] allow an application to explicitly map its GPU execution stream context and pass
it to the MPI library. This enables the MPI implementation to operate directly on GPU streams, thereby eliminating
unnecessary synchronization overhead and improving the performance of GPU-to-GPU communication. However,
none of these solutions have yet been incorporated into the official MPI standard.

4.4.2    Host vs Device-side API Both GPU-aware MPI and GPUCCL utilize host-side APIs, mandating that communi-
cation routines and their parameters (like size and data type) are expressed on the host CPU using host arguments.
GPUSHMEM offers both host and device-side APIs, enabling communication to be invoked directly from a GPU device
kernel with input arguments defined as device variables. This device-centric approach allows GPUSHMEM to better
handle low-latency and dynamic communication patterns. Data transfers are initiated within the computation kernel
and sent straight to the network, minimizing latency and making the approach ideal for fine-grained communication.
   However, this device-side invocation introduces programming and performance measurement challenges. Mixing
computation and communication within a single kernel complicates the programming model, and since current profiling
tools only operate at the kernel granularity, performance profiling is non-trivial. Moreover, especially when combined
with persistent kernels, the technique causes resource contention as streaming processors and threads must be shared
between the two. When employing techniques like dividing thread blocks (TBs) for overlap, synchronizing the dedicated
communication TBs versus the computation TBs becomes difficult. While Thread Block Clusters (TBCs) can potentially
mitigate this synchronization issue, their programming further complicates the overall programming complexity.
   Mimicking NVSHMEM, NVIDIA NCCL 2.28 [106] introduced a new device-side API to enable fusing communication
and compute. This was a significant departure from earlier versions where all NCCL operations were host-initiated.
The new API allows GPU kernels to directly initiate data movement, and its use requires setting up data buffers with
symmetric memory windows to facilitate direct GPU-to-GPU communication.

  1     //Host-side API
  2     if (rank == 0) { // process 0
  3         for (int j = 0; j < window_size; ++j) {
  4             MPI_Isend(send_buf, message_size, MPI_FLOAT, 1, 100, MPI_COMM_WORLD, send_request + j);
  5         }
  6         MPI_Waitall(window_size, send_request, reqstat);
  7         MPI_Recv(recv_buf, 1, MPI_FLOAT, 1, 101, MPI_COMM_WORLD, reqstat);
  8     } else { // process 1
  9         for (int j = 0; j < window_size; ++j) {
 10             MPI_Irecv(recv_buf, message_size, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, recv_request + j);
 11         }
 12         MPI_Waitall(window_size, recv_request, reqstat);
 13         MPI_Send(send_buf, 1, MPI_FLOAT, 0, 101, MPI_COMM_WORLD);
 14     }

                                 Listing 1. Simple one-way bandwidth benchmark with GPU-aware MPI




4.4.3    Programming Examples Listings 1, 2, 3 present simplified one-way bandwidth benchmarks using GPU-
aware MPI, NCCL, and device-side NVSHMEM. These examples highlight the semantic differences among the three
communication libraries. While all three take a buffer pointer and size, MPI and NCCL rely on a two-sided communication
model with synchronous send/receive semantics. In contrast, NVSHMEM employs a one-sided model: put/get operations
are asynchronous with respect to the remote GPU. This distinction is visible in Listings 1 and 2 versus Listing 3, where
NVSHMEM requires the sender to specify the receiver’s buffer address directly.
                                                                                                          Manuscript submitted to ACM
22                                                                                                                              D. Unat et al.

  1     //Host-side API
  2     if (rank == 0) { // process 0
  3         ncclGroupStart();
  4         for (int j = 0; j < window_size; ++j) {
  5             ncclSend(send_buf, message_size, ncclFloat, 1, comm, stream);
  6         }
  7         ncclGroupEnd();
  8         ncclRecv(recv_buf, 1, ncclFloat, 1, comm, stream);
  9     } else { // process 1
 10         ncclGroupStart();
 11         for (int j = 0; j < window_size; ++j) {
 12             ncclRecv(recv_buf, message_size, ncclFloat, 0, comm, stream);
 13         }
 14         ncclGroupEnd();
 15         ncclSend(send_buf, 1, ncclFloat, 0, comm, stream);
 16     }

                                       Listing 2. Simple one-way bandwidth benchmark with NCCL



  1     //Device-side API
  2     __global__ void comm_kernel_send(...) {
  3         nvshmemx_putmem_signal_nbi_block(recv_buf, send_buf, nx, signal_buf + blockIdx.x, 1, NVSHMEM_SIGNAL_ADD, 1);
  4         grid.sync();
  5         if (blockIdx.x == 0 && threadIdx.x == 0)
  6             nvshmem_signal_wait_until(signal_buf, NVSHMEM_CMP_GE, i + 1);
  7     }
  8
  9     __global__ void comm_kernel_recv(...) { // process 1
 10         if (threadIdx.x == 0)
 11             nvshmem_signal_wait_until(signal_buf + blockIdx.x, NVSHMEM_CMP_EQ, i + 1);
 12         grid.sync();
 13         if (blockIdx.x == 0)
 14             nvshmemx_putmem_signal_nbi_block(recv_buf, send_buf, 1 * sizeof(real), signal_buf, 1, NVSHMEM_SIGNAL_ADD, 0);
 15     }
 16     //Host-side API
 17     if (rank == 0) {// process 0
 18         nvshmemx_collective_launch(comm_kernel_send, dim3(window_size), dim3(1024), kernelArgs, 0, stream);
 19     } else { // process 1
 20         nvshmemx_collective_launch(comm_kernel_recv, dim3(window_size), dim3(1), kernelArgs, 0, stream);
 21     }

                            Listing 3. Simple one-way bandwidth benchmark with Device-Side GPUSHMEM



     MPI does not expose a stream parameter as discussed in Section 4.4.1, whereas NCCL and NVSHMEM operations are
explicitly tied to a CUDA stream. NCCL also supports grouping multiple operations between groupStart/groupEnd
to amortize launch overhead. Because MPI lacks stream semantics, MPI programs must explicitly synchronize CPU
and GPU progress e.g., using Waitall to ensure completion. NVSHMEM kernels that invoke device-side NVSHMEM
synchronization or collective APIs (e.g., nvshmem_wait, nvshmem_barrier, or collective operations) must be launched
with nvshmemx_collective_launch. Otherwise, behavior is undefined. Because this mechanism relies on CUDA
cooperative launch, the grid size is constrained so that the participating thread blocks can execute concurrently on the
GPU. If a kernel uses only one-sided put/get operations and does not use device-side NVSHMEM synchronization or
collective operations, a regular kernel launch is permitted.
     Supporting multiple communication libraries often forces developers to reimplement communication backends,
reducing productivity. To address this, Sagbili et al. proposed a unified communication interface that consolidates these
models under a single API and demonstrated performance comparable to naive implementations [126].

4.4.4    Performance Several works compare the performance of different user-level libraries, either on real applications
or through microbenchmarks. The authors of [147] implemented standard and pipelined Conjugate Gradient (CG)
using MPI, NCCL/RCCL, and NVSHMEM on both NVIDIA and AMD GPUs. They found that avoiding CPU-GPU
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                               23


synchronization via streams improved CG performance significantly by 5–15%. For NVIDIA systems, they suggested
using NCCL for small-message AllReduce but using MPI for point-to-point. On AMD GPUs, MPI outperformed vendor
solutions, which was attributed to the less mature vendor software.
   The benefits of NVSHMEM’s GPU-initiated communication have been demonstrated in molecular dynamics simula-
tions using GROMACS [42], yielding performance improvements of up to 2x over MPI. These gains are attributed to
the suitability of the PGAS model, which removes the CPU from the critical path and enables implementation-level
optimizations such as kernel fusion. However, with the recent introduction of the NCCL Device API, new studies suggest
that its performance is now comparable to NVSHMEM for both point-to-point and all-to-all communication.
   Similar advantages have been observed in graph processing applications, specifically Breadth-First Search (BFS) [118].
In these workloads, the inherent irregular and fine-grained access patterns align more effectively with the NVSHMEM
programming model than with standard MPI. Similar findings have been reported for other irregular workloads such as
updates of distributed Bloom filters and the search of connected components in large-scale graphs [82].
   Other works [40] compare NCCL (and RCCL) to GPU-Aware MPI on both collective and point-to-point operations
on three supercomputers. The results show a tendency for MPI to perform better in point-to-point operations, while
*CCL performs better in collectives. However, this also depends on the specific system and optimizations provided by
those libraries, and sometimes the best solution depends both on the number of nodes and vector size. Moreover, the
paper reports instability in RCCL at high node count, as also highlighted in a comparison between Cray MPICH and
RCCL performance on the Frontier supercomputer [140].
   Another limitation reported for both Open MPI [40] and Cray MPICH [140] concerns the execution of reduction-based
collectives, such as MPI_Allreduce and MPI_Reduce_scatter. Unless specifically configured with accelerator-offload
components, these MPI implementations often default to a host-staging protocol: data is copied to the host, reduced using
the CPU, and then copied back to the device. This round-trip data movement and reliance on CPU arithmetic capabilities
significantly degrade performance compared to native GPU collective libraries, which perform reductions directly on
the GPU. This was also addressed in a study optimizing the MFDn nuclear configuration interaction code [6]. The paper
demonstrated that replacing the baseline MPI/OpenACC implementation with native CUDA kernels for computation and
high-performance communication protocols yielded substantial gains. For large, network-bandwidth-bound problem
sizes, the NCCL/CUDA approach proved most effective, achieving up to a 4.9x speedup over the baseline by eliminating
the host-staging for reductions and fully leveraging GPU-native communication and parallelism.
   Despite these variations in programming and performance, all three libraries are mature, with diverse implementations
available; however, new GPUCCL variants as discussed in Section 4.2.2 are actively emerging from industry and research
to address evolving AI application needs.

4.4.5   Library Interactions Figure 5 illustrates the software stack hierarchy from the application down to the hardware.
Applications typically interface directly with high-level libraries such as MPI, GPUSHMEM, or GPUCCL. Lower-level
communication libraries, such as UCX or libfabric, generally serve as middleware used by these high-level frameworks
rather than being accessed directly by applications. While legacy MPI implementations often ran directly over transport
interfaces (e.g., libfabric or libibverbs), modern GPU-aware MPI stacks are more modular. They typically layer on top
of UCX for point-to-point operations and UCC for collectives. In some cases, such as with MVAPICH, MPI may also
offload collective operations directly to GPUCCL to leverage its topology-aware optimizations.
   Similarly, GPUSHMEM adopts a hybrid architecture: it utilizes UCX or direct transport interfaces (libfabric/libibverbs)
for low-latency point-to-point operations, while layering on top of GPUCCL to accelerate collective routines. Finally,
                                                                                                 Manuscript submitted to ACM
24                                                                                                               D. Unat et al.


                                                         Application



                              MPI                                                           GPUSHMEM

                                                           GPUCCL

                              UCC



                                                             UCX



                                      libfabric, libibverbs, sockets, shared mem., etc...



                                  Hardware (NVLink, InfinityFabric, NVSwitch, UALink, etc...)


         Fig. 5. Interactions between different GPU-GPU communication libraries. Figure is available under CC-BY [150].




GPUCCL typically interfaces directly with the transport layer (libfabric or libibverbs), though it can also be configured
to run on top of UCX via specific plugins for enhanced portability. Last, UCC serves as a higher-level abstraction,
executing collective operations by either offloading them to GPUCCL or employing its own implementations over UCX
and low-level transports.


5     Discussion, Challenges, and Outlook
As multi-GPU execution becomes a necessity for many real-world workloads, we dedicate this section to discussions and
outlooks, presenting what we believe to be fertile ground for current and future research on GPU-centric communication.


5.1    Moving Away from CPU
There are several reasons why we believe moving away from CPU-controlled execution shows promise. First, it
addresses the issue of CPU-induced latency barriers that are caused by kernel launch and memory copy overheads.
These barriers become more significant in strong scaling scenarios as the number of GPUs increases and computation
per GPU decreases. In latency-bound settings, traditional CPU-controlled implementations are not able to overlap
communication with computation as the latencies to initiate the operations take longer than the operations themselves
effectively serializing communication and computation. On the other hand, CPU-free execution can still achieve
adequate levels of overlap even when latencies dominate [61, 154]. Second, the parallelism offered by within-kernel
communication is well-suited for persistent kernels to take advantage of. With the use of efficient communication and
synchronization APIs, this execution model can achieve higher bandwidths and lower latencies than CPU-controlled
implementations. Additionally, CPU-free execution, by virtue of inlining communication with computation, is well-
suited for applications with fine-grained communication [27]. Third, ROC_SHMEM allows for the first time to fully
migrate application execution to the device with zero reliance on CPU helper threads. NVSHMEM with its IBGDA
transport can also migrate a significant amount of execution to GPU but must still rely on the CPU for functional
correctness.
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                            25


  However, there are several challenges facing this model of execution. One major challenge is that persistent kernels
can result in reduced occupancy potentially bottlenecking computation. As of today, if global device or multi-GPU
barriers are required, persistent kernels must be launched in a cooperative manner. This means that only as many
threads can be launched as can run concurrently at the same time making hardware oversubscription impossible. As
a result, workload decomposition and scheduling, which were previously handled by the hardware scheduler, now
need to be manually done by the programmer. This manual approach is unlikely to be as efficient as hardware-based
scheduling, and compute-intensive applications are likely to suffer. Furthermore, long-running persistent kernels will
consume more registers and may use shared memory as well limiting occupancy even further. Nevertheless, we see a
number of solution to this problem. First, there is a large amount of high bandwidth shared memory available across
application execution, which can potentially nullify the performance hit caused by reduced occupancy. Second, we
predict that as GPU vendors strive more and more for greater GPU autonomy, they will introduce APIs that allow for
combining persistent kernels with hardware oversubscription. The manual decomposition can also be handled by an
optimized compiler / runtime system. Alternatively, collective operations can be offloaded to the network components
[143], freeing the resources in the GPU to compute.
  Another potential problem with NVSHMEM and ROC_SHMEM is their ease of integration into existing runtimes.
Both libraries center around a symmetric heap and all communication buffers must be allocated collectively on the same
symmetric heap by all GPUs. This symmetric allocation requires library-specific allocators. Existing runtimes may find
it hard to add support for NVSHMEM and ROC_SHMEM because of the symmetric memory allocation requirement.

5.2   UCX as a Potential Pathway for GPU-Awareness
Unified Communication X (UCX) is an open-source communication framework that abstracts over several network
APIs, programming models, protocols, and implementations. The idea is to provide a set of high-level primitives while
hiding the low-level implementation details behind the UCX runtime. UCX is organized into three main components:
UCP (UC-Protocol), which provides high-level communication primitives such as messaging and remote memory
access; UCT (UC Transport), which offers low-level access to network hardware; and UCS (UC Services), which supplies
common utilities for memory management and threading.
  At runtime, a UCP layer dynamically selects the appropriate UCT transport based on the pointer type and system
topology. For example, when a CUDA pointer is passed to an inter-node UCP communication, the rc_mlx5 transport
may be chosen, whereas for intra-node communication, the cuda_ipc transport is typically preferred. As a result, UCP
can accept device pointers as data payloads and perform transfers using the most suitable mechanism, making UCX
inherently GPU-aware. This GPU-awareness of UCX allows higher-level communication libraries such as OpenMPI
and MPICH to handle device pointers directly in their communication primitives. For example in CUDA-aware MPI,
UCX eliminates explicit host memory copies by enabling GPU memory pointers to be passed directly to MPI functions
[133, 144].
  Using UCX for realizing GPU-centric communication is a recent direction in the literature that has started to take
hold. Perhaps the most relevant example is that of ROCm-awareness for MPI implementations. Much early work on
GPU-aware MPI was done for NVIDIA GPUs using native CUDA libraries and then integrated directly into MPI runtimes.
Perhaps reticent to replicate the same work to make their runtimes ROCm-aware, most MPI implementations provide
ROCm-awareness only through UCX. OpenMPI and MPICH additionally provide CUDA support through UCX, besides
their native integration. In non-MPI work, Choi et al. extend the UCX layer in Charm++ to provide GPU-aware
communication for several programming models in the Charm++ ecosystem [28, 29].
                                                                                              Manuscript submitted to ACM
26                                                                                                        D. Unat et al.


     We predict that more runtimes will gravitate toward UCX to add support for GPU-centric communication. Using UCX
APIs frees the programmers from relying on native vendor-specific APIs and allows adding GPU-aware communication
for both ROCm and CUDA. A performance comparison between GPU-awareness through native APIs and that provided
through UCX would be helpful. In one work in this direction, Khorassani et al. provide a native ROCm-aware runtime
for MVAPICH2, which outperforms OpenMPI with UCX on a cluster of AMD GPUs [129]. However, the performance
difference may be due to differences in the MPI implementations, not UCX. A current limitation of UCX is that it is
always on the host, and it does not use any communication kernel, but relies on the traditional cudamemcpy family of
functions in addition to zero copy RDMA.

5.3     CPU-Free Networking
As the trend toward GPU-centric communication and greater GPU autonomy continues to accelerate, several works
have suggested migrating most or all of the networking stack to the kernel. This is typically done by launching a single
long-running persistent kernel and moving both the data and control paths to the GPU.
     In the earliest work, GGAS [107] proposes changes to network devices to implement a unified global address space
that allows moving the control path entirely to the GPU. This is accomplished by using a persistent kernel that contains
the computation, communication, and synchronization all on the device-side. While the work was the first of its kind
and showed performance improvements compared to a CUDA-Aware MPI baseline, the experiments were conducted on
two GPU nodes with one GPU each, while the proposed hardware changes were emulated on an FPGA. Follow-up work
showed that GGAS, by virtue of eliminating CPU involvement in the control path, can achieve further performance
improvements and reduce energy usage compared to CPU-controlled baselines [66, 109].
     Several more works on CPU-free networking followed. Oden et al. [108] use GPUDirect RDMA to allow GPUs to
directly interface with Infiniband network devices without the involvement of the host CPU. They do this by mapping the
entire Infiniband context to the device-side and using the GPU to generate and send work requests to the HCA. However,
because of slow single-thread work request generation performance on the GPU, the proposed changes deteriorated
performance compared to CPU-controlled baselines. Follow-up work ameliorated these performance limitations and
showed much more promising results [65, 67]. Another work combines the proposed GPU-side Infiniband Verbs
with CUDA Dynamic Parallelism to optimize the bottleneck of intra-kernel synchronization [110]. GPUrdma also
implements Infiniband on the GPU and proposes a GPU-side library for direct communication from within persistent
GPU kernels with zero CPU involvement. The proposed design outperforms a CPU-controlled baseline on a series of
microbenchmarks but runs into correctness issues resulting from the use of persistent kernels and GPU-NIC interaction
[37].
     Silberstein et al. implement GPUNet, which provides GPU-side socket abstractions and networking primitives [139].
GPUNet allows invoking the communication on the GPU but does not fully migrate the control path to the device;
instead, it relies on CPU helper threads to perform the actual communication. A similar approach is adopted by dCUDA,
which provides device-side APIs with MPI semantics but translates them to standard MPI calls performed by CPU
helper threads [49]. LeBeane et al. categorized GPU networking methods discussing at length their deficiencies. In
response, they propose GPU-TN, a NIC hardware mechanism that allows the CPU to create and register messages with
the NIC and the GPU to trigger them from a running persistent kernel [73]. Another work, ComP-Net, uses embedded
GPU microprocessors to offload helper threads from the CPU to the GPU [74]. While both GPU-TN and ComP-Net
show promising performance, they require hardware changes to the NIC and the GPU and, thus, rely on simulation to
obtain results.
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                               27


5.4   Broader GPU Autonomy
The recent proliferation of GPU-centric communication represents the general trend toward broader GPU autonomy.
Several works, early and recent, have tried to hand the GPU the reins of domains that have traditionally been the
purview of the CPU. In an early work, Stuart et al. propose methods that allow the GPU to issue callbacks to the CPU
[142]. Silberstein et al. implement GPUfs, which allows the GPU to request files on the host CPU directly from inside a
GPU kernel [138]. Veselý et al. implement support for invoking POSIX system calls from inside GPU kernels through
changes to the Linux kernel [153]. NVIDIA’s GPUDirect Storage provides a direct data path between GPUs and storage
but still relies on the CPU to orchestrate execution [145].
  SmartIO enables low-cost, efficient I/O disaggregation across PCIe-connected hosts by allowing remote machines to
borrow and directly access devices such as NVMe, GPUs, and NICs as if they were local. Building on this idea, Qureshi
et al. at NVIDIA present BaM allowing GPUs to directly access storage without any CPU involvement. Experimental
results show that BaM outperforms GPUDirect Storage on several workloads [124]. Turimbetov et al. showed CPU-free
device-side task graph execution on multiple GPUs, eliminating kernel launch and scheduling overheads from the
host-side [148]. Lastly, Baydamirli et al. [17] presented a compiler support for autonomous execution for multi-GPU
systems, enabling GPU-initiated communication within Python code using NVSHMEM.
  These and other works show a clear trend toward general GPU autonomy. In line with this, we expect further opti-
mizations to GPU-centric communication. Several recent mechanisms are promising, as pointed out by Punniyamurthy
et al. [122]. First, the recent thread block (TB) cluster abstraction could benefit device-side communication-computation
overlap and inter-TB synchronization. Second, AMD’s recent cache flush instructions allow flushing the cache before
initiating network communication, meaning communication no longer needs to be allocated as uncacheable. Third,
recent hardware trends like fatter GPU nodes and tight GPU-NIC integration are also promising [8, 44]. For example,
the most recent iteration of NVSwitch directly connects 256 Grace Hopper Superchips enabling direct P2P all-to-all
communication at an unprecedented scale.
  While these advances push the system toward broader GPU autonomy, the CPU plays a complementary role to
provide system-level responsibilities. Debugging, profiling, monitoring, and software development still fundamentally
rely on the CPU’s broader visibility into the system, its mature tooling ecosystem, and its tight integration with
operating system services.


5.5   Collective Algorithms Design
Multi-GPU systems present unprecedented challenges in the design of collective operations algorithms. First, GPUs
within the same node are often arranged in complex and non-traditional topologies, making existing collective algorithms
less effective. Second, there is significant heterogeneity in the bandwidth between GPU pairs, both within a node and
between nodes. For instance, there can be up to a 4x difference in bandwidth among different GPU pairs on the same
node [36], and up to a 10x gap between intra-node and inter-node bandwidth. Third, these topological and connectivity
characteristics frequently change between GPU generations. These factors complicate the design of collective algorithms,
potentially leading to underutilization of available bandwidth and poor performance of collective operations.
  Some approaches use linear programming formulations to find the optimal collective algorithm based on a network
specification and the size of the collective [23, 130, 155]. However, this involves solving an NP-hard problem that scales
exponentially. For example, finding a solution for 128 nodes can take up to 11 hours [130], and a new solution might
be needed if the number of nodes or the size of the collective changes. This complexity makes generating collective
                                                                                                 Manuscript submitted to ACM
28                                                                                                           D. Unat et al.


algorithms for large systems challenging, if not impossible. To simplify the implementation of multi-GPU collective
operations, MSCCL [35] provides a high-level language for expressing collective operations, which is then compiled to
NCCL code.


5.6     Debugging, Profiling, Benchmarking Support
Efficient programming tools are essential for productive multi-GPU programming. However, the available tools are
severely lacking when it comes to communication native to the GPU. While NVIDIA’s flagship system-level profiling
tool, NSight Systems, provides a detailed view of host-controlled communication, it falls short in providing information
on device-native communication, including Direct Load/Store P2P communication and communication induced by
libraries such as NCCL and NVSHMEM.
     Palwisha et al. present ComScribe [5], a tool that allows monitoring NCCL collective and P2P communication.
However, it is limited to a single node and relies on the deprecated nvprof tool. New efforts in this area have been
made by Snoopie [62], which focuses on profiling and visualizing GPU-centric communication. Snoopie is capable of
attributing communication to source code lines and objects involved in communication, offering different levels of
granularity. This allows for both a coarse-grained overview of the system and a detailed view of a specific object or
device in terms of data movement. Similarly, ucTrace [46] is a multi-node profiling and visualization framework for the
UCX communication layer. It exposes communication behavior across multiple layers of the software and hardware
stack.
     Another class of tools that can help with debugging GPU communication are race detectors. Given that many of
the aforementioned communication libraries use a partitioned global address space model compared with message
passing, there is a high likelihood of introducing race hazards. Race detector tools are vital to navigate the complexities
of shared data in multi-GPU programming. Despite their importance, none of the existing tools are capable of detecting
race hazards in multi-GPU programming. Compute Sanitizer’s Racecheck tool [86] is limited to on-chip shared memory,
supporting the detection of race hazards only within a single GPU context. HiRace [63] improves on that by supporting
global memory and can detect many types of race hazards that Compute Sanitizer’s Racecheck cannot, but it is also
limited to a single GPU context.
     Several benchmarking tools have been proposed for measuring the performance of GPU-GPU communication.
Standard suites such as the *ccl-tests for NCCL and RCCL are commonly used to assess the performance of both
collectives and point-to-point communication [15, 105]. Similarly, the OSU benchmark suite (OMB) provides extensive
support for GPU-aware MPI environments [21]. NVSHMEM also provides specific micro-benchmarks to evaluate
device-initiated put and get latency. However, these tools typically report performance measurements aggregated
across multiple iterations, which masks critical network performance variability and transient noise [39]. Furthermore,
they often lack baselines for explicit peer-to-peer copies. To address these limitations, the Blink-GPU benchmark was
proposed [40, 70], enabling the capture of per-iteration timing to expose performance anomalies that aggregated metrics
miss.
     We believe that introducing debugging and profiling tools capable of detecting fine-grained device-native transfers
both within and across nodes, in addition to race detectors capable of capturing race hazards in a multi-GPU context, is
crucial for further advancements in GPU-centric communication.


5.7     Compression-Accelerated GPU Communication

Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                                29


The high cost of moving large data volumes across the network, particularly for inter-node GPU communication, has
motivated the development of techniques for compressing data prior to transmission. While this approach introduces
computational overhead for compression and decompression, this cost is frequently amortized by the substantial
reduction in communication time. To maximize the data reduction (i.e., achieve a high compression ratio), lossy
compression is typically employed. Specifically, error-bounded lossy compression [41] allows for high compression
ratios while guaranteeing that data distortion remains within a user-defined accuracy threshold. This technique has been
integrated into collective communication libraries for both CPU and GPU architectures [57–59, 69, 167], demonstrating
significant speedups over conventional libraries like MPI and NCCL.
    While effective at reducing data volume, these first-generation solutions were fundamentally limited by the
Decompression-Operation-Compression (DOC) workflow. This workflow mandates that data be fully decompressed
before any computation (such as a reduction in an Allreduce operation) can be performed, only to be immediately
recompressed for subsequent steps, incurring significant processing overhead. This limitation has motivated the devel-
opment of homomorphic compression, which enables computation to be performed directly on compressed data. Recent
approaches have successfully implemented this paradigm for both CPU [59] and GPU [57] architectures. By allowing
GPUs to compute and communicate entirely in the compressed domain, this approach represents the most advanced
solution, maximizing throughput by addressing both the data volume bottleneck and the internal DOC processing
overhead.


5.8    Level of Maturity and User-Level Perspective
It is crucial to clarify the role and maturity of the technologies that underpin modern accelerator communication, such
as NVIDIA’s GPUDirect, GPUDirect RDMA, AMD’s ROCm peer-to-peer capabilities, and GDRCopy. These components
are not high-level, user-facing APIs, nor are they hacks. Rather, they are foundational, vendor-supported hardware and
driver-level functionalities. They represent the official, low-level architecture integrated directly into the accelerator’s
core runtime environment and the system’s kernel drivers.
    The maturity and reliability of these solutions are demonstrated by their universal adoption as the essential building
blocks for the entire high-performance, open-source communication ecosystem. High-level libraries, including all major
MPI implementations (e.g., Open MPI, MVAPICH, Intel MPI), low-level communication middleware (e.g., UCX), and
vendors’s own *CCL, are built on top of these APIs. These libraries explicitly use those lower-level functionalities to
enable their GPU-aware communication paths. Therefore, the benefits of these technologies (namely, the elimination
of the CPU as a bottleneck and the ability to achieve true, zero-copy, low-latency communication by bypassing host
memory) are the well-documented, stable, fundamental mechanisms that enable all modern multi-node, multi-GPU
supercomputing.


6     Conclusion
Traditionally, multi-GPU communication was managed by the CPU, but recent advancements in GPU-centric communi-
cation have begun shifting this responsibility, allowing GPUs to take more control over the communication tasks. This
paper provides an in-depth exploration of GPU-centric communication, emphasizing vendor-supplied mechanisms and
user-level library support. It seeks to demystify the complexities and variety of options in this area, define key terms,
and categorize the various approaches used within individual nodes and across multiple nodes.
                                                                                                  Manuscript submitted to ACM
30                                                                                                                                      D. Unat et al.


     The discussion includes an analysis of vendor-provided communication and memory management mechanisms
for multi-GPU execution, as well as a review of major communication libraries such as CUDA-aware MPI, NCCL/RC-
CL/oneCCL, NVSHMEM, ROC_SHMEM, and Intel SHMEM highlighting their advantages, challenges, and performance
considerations. Furthermore, the paper presents important research paradigms such as CPU-free networking, debug-
ging tools for communication, and discusses future directions, and unresolved questions. By thoroughly examining
GPU-centric communication techniques across both software and hardware layers, this paper aims to equip researchers,
developers, engineers, and library designers with the knowledge needed to fully leverage multi-GPU systems.


Acknowledgment
Authors at Koç University were supported from the European Research Council (ERC) under the European Union’s
Horizon 2020 research and innovation programme (grant agreement No 949587). Flavio Vella and Daniele De Sensi are
supported by the European Union’s Horizon Europe under grant 101175702 (NET4EXA). Daniele De Sensi is supported
by Sapienza University Grants ADAGIO and D2QNeT (Bando per la ricerca di Ateneo 2023 and 2024).


References
  [1] Elena Agostini. 2023. Inline GPU Packet Processing with NVIDIA DOCA GPUNetIO. https://developer.nvidia.com/blog/inline-gpu-packet-
      processing-with-nvidia-doca-gpunetio/.
  [2] Elena Agostini. 2023. Optimizing Inline Packet Processing Using DPDK and GPUDirect with GPUs. https://developer.nvidia.com/blog/optimizing-
      inline-packet-processing-using-dpdk-and-gpudev-with-gpus/.
  [3] Elena Agostini, Davide Rossetti, and Sreeram Potluri. 2017. Offloading Communication Control Logic in GPU Accelerated Applications. In
      Proceedings of the 17th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (Madrid, Spain) (CCGrid ’17). Institute for Electrical
      and Electronics Engineers, New York, NY, USA, 248–257. https://doi.org/10.1109/CCGRID.2017.29
  [4] E. Agostini, D. Rossetti, and S. Potluri. 2018. GPUDirect Async: Exploring GPU synchronous communication techniques for InfiniBand clusters. J.
      Parallel and Distrib. Comput. 114 (2018), 28–45. https://doi.org/10.1016/j.jpdc.2017.12.007
  [5] Palwisha Akhtar, Erhan Tezcan, Fareed Mohammad Qararyah, and Didem Unat. 2021. ComScribe: Identifying Intra-node GPU Communication. In
      Benchmarking, Measuring, and Optimizing, Felix Wolf and Wanling Gao (Eds.). Springer International Publishing, Cham, 157–174.
  [6] Abdullah Alperen, Nan Ding, Khaled Z. Ibrahim, Pieter Maris, Leonid Oliker, Chao Yang, and Hasan Metin Aktulga. 2025. Optimizing Nuclear
      Configuration Interaction Calculations on GPUs: A Comparative Performance Study of Programming Models. In ISC High Performance 2025
      Research Paper Proceedings (40th International Conference). 1–12.
  [7] AMD. [n. d.]. AMD Instinct MI200 Instruction Set Architecture. https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-
      set-architecture.pdf.
  [8] AMD. 2021. AMD CDNA™ 2 ARCHITECTURE. https://www.amd.com/system/files/documents/amd-cdna2-white-paper.pdf.
  [9] AMD. 2023. GPU-aware MPI with ROCm. https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-gpu-aware-mpi-readme/#.
 [10] AMD. 2023. ROCK-Kernel-Driver. https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver.
 [11] AMD. 2023. ROCm Documentation: GPU-Enabled MPI. https://rocm.docs.amd.com/en/latest/how_to/gpu_aware_mpi.html.
 [12] AMD. 2023. ROCnRDMA. https://github.com/rocmarchive/ROCnRDMA.
 [13] AMD. 2023. ROC_SHMEM. https://github.com/ROCm-Developer-Tools/ROC_SHMEM.
 [14] AMD. 2025. RCCL Developer Guide. https://rocm.docs.amd.com/projects/rccl/en/latest/. Accessed: 2025-02-20.
 [15] AMD. 2025. RCCL Tests. https://github.com/ROCm/rccl-tests.
 [16] Dip Sankar Banerjee, Khaled Hamidouche, and Dhabaleswar K. Panda. 2016. Designing High Performance Communication Runtime for GPU
      Managed Memory: Early Experiences (GPGPU ’16). Association for Computing Machinery, New York, NY, USA, 82–91. https://doi.org/10.1145/
      2884045.2884050
 [17] Javid Baydamirli, Tal Ben Nun, and Didem Unat. 2024. Autonomous Execution for Multi-GPU Systems: Compiler Support. In SC24-W: Workshops of
      the International Conference for High Performance Computing, Networking, Storage and Analysis. 1129–1140. https://doi.org/10.1109/SCW63240.
      2024.00155
 [18] Tal Ben-Nun, Michael Sutton, Sreepathi Pai, and Keshav Pingali. 2020. Groute: Asynchronous Multi-GPU Programming Model with Applications
      to Large-Scale Graph Processing. ACM Trans. Parallel Comput. 7, 3, Article 18 (jun 2020), 27 pages. https://doi.org/10.1145/3399730
 [19] Massimo Bernaschi, Elena Agostini, and Davide Rossetti. 2021.              Benchmarking multi-GPU applications on modern multi-GPU in-
      tegrated systems.        Concurrency and Computation: Practice and Experience 33, 14 (2021), e5470.             https://doi.org/10.1002/cpe.5470
      arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/cpe.5470
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                                                          31


[20] Alex Brooks, Philip Marshall, David Ozog, Md. Wasi-Ur-Rahman, Lawrence Stewart, and Rithwik Tom. 2024. IntelSHMEM: GPU-initiated
     OpenSHMEM using SYCL. In SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis.
     1288–1301. https://doi.org/10.1109/SCW63240.2024.00169
[21] D. Bureddy, H. Wang, A. Venkatesh, S. Potluri, and D. K. Panda. 2012. OMB-GPU: a micro-benchmark suite for evaluating MPI libraries on
     GPU clusters. In Proceedings of the 19th European Conference on Recent Advances in the Message Passing Interface (Vienna, Austria) (EuroMPI’12).
     Springer-Verlag, Berlin, Heidelberg, 110–120. https://doi.org/10.1007/978-3-642-33518-1_16
[22] Idan Burstein. 2021. Nvidia Data Center Processing Unit (DPU) Architecture. In 2021 IEEE Hot Chips 33 Symposium (HCS). 1–20. https:
     //doi.org/10.1109/HCS52781.2021.9567066
[23] Zixian Cai, Zhengyang Liu, Saeed Maleki, Madanlal Musuvathi, Todd Mytkowicz, Jacob Nelson, and Olli Saarikivi. 2021. Synthesizing Optimal
     Collective Algorithms. In Proceedings of the 26th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (Virtual Event,
     Republic of Korea) (PPoPP ’21). Association for Computing Machinery, New York, NY, USA, 62–75. https://doi.org/10.1145/3437801.3441620
[24] Chen-Chun Chen, Kawthar Shafie Khorassani, Goutham Kalikrishna Reddy Kuncham, Rahul Vaidya, Mustafa Abduljabbar, Aamir Shafi, Hari
     Subramoni, and Dhabaleswar K. Panda. 2023. Implementing and Optimizing a GPU-aware MPI Library for Intel GPUs: Early Experiences. In 2023
     IEEE/ACM 23rd International Symposium on Cluster, Cloud and Internet Computing (CCGrid). 131–140. https://doi.org/10.1109/CCGrid57682.2023.
     00022
[25] Chen-Chun Chen, Goutham Kalikrishna Reddy Kuncham, Pouya Kousha, Hari Subramoni, and Dhabaleswar K. Panda. 2024. Design and
     Implementation of an IPC-based Collective MPI Library for Intel GPUs. In Practice and Experience in Advanced Research Computing 2024:
     Human Powered Computing (Providence, RI, USA) (PEARC ’24). Association for Computing Machinery, New York, NY, USA, Article 17, 9 pages.
     https://doi.org/10.1145/3626203.3670549
[26] Chen-Chun Chen, Kawthar Shafie Khorassani, Pouya Kousha, Qinghua Zhou, Jinghan Yao, Hari Subramoni, and Dhabaleswar K. Panda. 2023.
     MPI-xCCL: A Portable MPI Library over Collective Communication Libraries for Various Accelerators. In Proceedings of the SC ’23 Workshops of the
     International Conference on High Performance Computing, Network, Storage, and Analysis (Denver, CO, USA) (SC-W ’23). Association for Computing
     Machinery, New York, NY, USA, 847–854. https://doi.org/10.1145/3624062.3624153
[27] Yuxin Chen, Benjamin Brock, Serban Porumbescu, Aydın Buluç, Katherine Yelick, and John D. Owens. 2022. Scalable Irregular Parallelism with
     GPUs: Getting CPUs out of the Way. In Proceedings of the International Conference on High Performance Computing, Networking, Storage and
     Analysis (Dallas, Texas) (SC ’22). Institute for Electrical and Electronics Engineers, New York, NY, USA, Article 50, 16 pages.
[28] Jaemin Choi, Zane Fink, Sam White, Nitin Bhat, David F. Richards, and Laxmikant V. Kale. 2021. GPU-aware Communication with UCX in Parallel
     Programming Models: Charm++, MPI, and Python. In 2021 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW).
     479–488. https://doi.org/10.1109/IPDPSW52791.2021.00079
[29] Jaemin Choi, Zane Fink, Sam White, Nitin Bhat, David F. Richards, and Laxmikant V. Kale. 2022. Accelerating communication for parallel
     programming models on GPU systems. Parallel Comput. 113 (2022), 102969. https://doi.org/10.1016/j.parco.2022.102969
[30] Jaemin Choi, David F. Richards, and Laxmikant V. Kale. 2021. CharminG: A Scalable GPU-Resident Runtime System. In Proceedings of the 30th
     International Symposium on High-Performance Parallel and Distributed Computing (Virtual Event, Sweden) (HPDC ’21). Association for Computing
     Machinery, New York, NY, USA, 261–262. https://doi.org/10.1145/3431379.3464454
[31] Ching-Hsiang Chu, Sreeram Potluri, Anshuman Goswami, Manjunath Gorentla Venkata, Neena Imam, and Chris J. Newburn. 2019. Designing
     High-Performance In-Memory Key-Value Operations with Persistent GPU Kernels and OpenSHMEM. In OpenSHMEM and Related Technologies.
     OpenSHMEM in the Era of Extreme Heterogeneity, Swaroop Pophale, Neena Imam, Ferrol Aderholdt, and Manjunath Gorentla Venkata (Eds.).
     Springer International Publishing, Cham, 148–164.
[32] Jan Ciesko. 2023. Kokkos Remote Spaces Repository. https://github.com/kokkos/kokkos-remote-spaces.
[33] Carsten Clauss. 2025. GitHub - Pull Request for adding support for Unified Collective Communication (UCC). https://github.com/pmodels/mpich/
     pull/7578. Accessed: 2025-11-02.
[34] NVIDIA Corporation. 2023. NVIDIA DOCA SDK Documentation. https://docs.nvidia.com/doca/sdk/index.html.
[35] Meghan Cowan, Saeed Maleki, Madanlal Musuvathi, Olli Saarikivi, and Yifan Xiong. 2023. MSCCLang: Microsoft Collective Communication
     Language. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems,
     Volume 2 (Vancouver, BC, Canada) (ASPLOS 2023). Association for Computing Machinery, New York, NY, USA, 502–514. https://doi.org/10.1145/
     3575693.3575724
[36] CSC. 2024. LUMI-G Supercomputer. https://docs.lumi-supercomputer.eu/hardware/lumig/.
[37] Feras Daoud, Amir Watad, and Mark Silberstein. 2016. GPUrdma: GPU-Side Library for High Performance Networking from GPU Kernels (ROSS
     ’16). Association for Computing Machinery, New York, NY, USA, Article 6, 8 pages. https://doi.org/10.1145/2931088.2931091
[38] Seth Howell Davide Rossetti, Pak Markthub. 2021. The Latest in GPUDirect. https://www.nvidia.com/en-us/on-demand/session/gtcspring21-
     s32039/.
[39] Daniele De Sensi, Salvatore Di Girolamo, and Torsten Hoefler. 2019. Mitigating Network Noise on Dragonfly Networks Through Application-aware
     Routing. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (Denver, Colorado) (SC
     ’19). ACM, New York, NY, USA, Article 16, 32 pages. https://doi.org/10.1145/3295500.3356196
[40] Daniele De Sensi, Lorenzo Pichetti, Flavio Vella, Tiziano De Matteis, Zebin Ren, Luigi Fusco, Matteo Turisini, Daniele Cesarini, Kurt Lust, Animesh
     Trivedi, Duncan Roweth, Filippo Spiga, Salvatore Di Girolamo, and Torsten Hoefler. 2024. Exploring GPU-to-GPU Communication: Insights into
                                                                                                                        Manuscript submitted to ACM
32                                                                                                                                    D. Unat et al.


      Supercomputer Interconnects. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis
      (SC’24). https://doi.org/10.1109/SC41406.2024.00039
 [41] Sheng Di, Jinyang Liu, Kai Zhao, Xin Liang, Robert Underwood, Zhaorui Zhang, Milan Shah, Yafan Huang, Jiajun Huang, Xiaodong Yu, Congrong
      Ren, Hanqi Guo, Grant Wilkins, Dingwen Tao, Jiannan Tian, Sian Jin, Zizhe Jian, Daoce Wang, Md Hasanur Rahman, Boyuan Zhang, Shihui Song,
      Jon Calhoun, Guanpeng Li, Kazutomo Yoshii, Khalid Alharthi, and Franck Cappello. 2025. A Survey on Error-Bounded Lossy Compression for
      Scientific Datasets. ACM Comput. Surv. 57, 11, Article 287 (June 2025), 38 pages. https://doi.org/10.1145/3733104
 [42] Mahesh Doijade, Andrey Alekseenko, Ania Brown, Alan Gray, and Szilárd Páll. 2025. Redesigning GROMACS Halo Exchange: Improving Strong
      Scaling with GPU-initiated NVSHMEM. In Proceedings of the SC ’25 Workshops of the International Conference for High Performance Computing,
      Networking, Storage and Analysis (SC Workshops ’25). ACM, 1314–1329. https://doi.org/10.1145/3731599.3767508
 [43] Nikoli Dryden, Naoya Maruyama, Tim Moon, Tom Benson, Andy Yoo, Marc Snir, and Brian Van Essen. 2018. Aluminum: An Asynchronous,
      GPU-Aware Communication Library Optimized for Large-Scale Training of Deep Neural Networks on HPC Systems. In 2018 IEEE/ACM Machine
      Learning in HPC Environments (MLHPC). 1–13. https://doi.org/10.1109/MLHPC.2018.8638639
 [44] Jonathon Evans, Michael Andersch, Vikram Sethi, Gonzalo Brito, and Vishal Mehta. 2022. NVIDIA Grace Hopper Superchip Architecture In-Depth.
      https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/.
 [45] Denis Foley and John Danskin. 2017. Ultra-Performance Pascal GPU and NVLink Interconnect. IEEE Micro 37, 2 (2017), 7–17. https://doi.org/10.
      1109/MM.2017.37
 [46] Emir Gencer, Mohammad Kefah Taha Issa, Ilyas Turimbetov, James D. Trotter, and Didem Unat. 2026. ucTrace: A Multi-Layer Profiling Tool for
      UCX-driven Communication. arXiv:2602.19084 [cs.DC] https://arxiv.org/abs/2602.19084
 [47] Taylor Groves, Ben Brock, Yuxin Chen, Khaled Z. Ibrahim, Lenny Oliker, Nicholas J. Wright, Samuel Williams, and Katherine Yelick. 2020.
      Performance Trade-offs in GPU Communication: A Study of Host and Device-initiated Approaches. In 2020 IEEE/ACM Performance Modeling,
      Benchmarking and Simulation of High Performance Computer Systems (PMBS). 126–137. https://doi.org/10.1109/PMBS51919.2020.00016
 [48] Yanfei Guo, Kenneth Raffenetti, Rob Latham, Marc Snir, and Hui Zhou. 2022. MPICH for Exascale. https://www.exascaleproject.org/wp-
      content/uploads/2022/06/2022_ECPAM_BoF_MPICH_ANL.pdf.
 [49] Tobias Gysi, Jeremia Bär, and Torsten Hoefler. 2016. dCUDA: Hardware Supported Overlap of Computation and Communication. In SC ’16:
      Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 609–620. https://doi.org/10.1109/SC.
      2016.51
 [50] Khaled Hamidouche, Ammar Ahmad Awan, Akshay Venkatesh, and Dhabaleswar K. Panda. 2016. CUDA M3: Designing Efficient CUDA Managed
      Memory-Aware MPI by Exploiting GDR and IPC. In 2016 IEEE 23rd International Conference on High Performance Computing (HiPC). 52–61.
      https://doi.org/10.1109/HiPC.2016.016
 [51] Khaled Hamidouche and Michael LeBeane. 2020. GPU INitiated OPenSHMEM: Correct and Efficient Intra-Kernel Networking for DGPUs. In
      Proceedings of the 25th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (San Diego, California) (PPoPP ’20). Association
      for Computing Machinery, New York, NY, USA, 336–347. https://doi.org/10.1145/3332466.3374544
 [52] Mark Harris. 2012. How to Optimize Data Transfers in CUDA C/C++. https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/.
 [53] Mert Hidayetoglu, Simon Garcia De Gonzalo, Elliott Slaughter, Yu Li, Christopher Zimmer, Tekin Bicer, Bin Ren, William Gropp, Wen-Mei Hwu,
      and Alex Aiken. 2024. CommBench: Micro-Benchmarking Hierarchical Networks with Multi-GPU, Multi-NIC Nodes. In Proceedings of the 38th
      ACM International Conference on Supercomputing (Kyoto, Japan) (ICS ’24). Association for Computing Machinery, New York, NY, USA, 426–436.
      https://doi.org/10.1145/3650200.3656591
 [54] Mert Hidayetoglu, Simon Garcia de Gonzalo, Elliott Slaughter, Pinku Surana, Wen-mei Hwu, William Gropp, and Alex Aiken. 2025. HiCCL:
      A Hierarchical Collective Communication Library. In 2025 IEEE International Parallel and Distributed Processing Symposium (IPDPS). 950–961.
      https://doi.org/10.1109/IPDPS64566.2025.00089
 [55] HPE. 2021. Cray MPICH Documentation. https://cpe.ext.hpe.com/docs/mpt/mpich/intro_mpi.html.
 [56] Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, James Dinan, Jeff Hammond, and Torsten Hoefler. 2025.
      Demystifying NCCL: An in-depth analysis of GPU communication protocols and algorithms. arXiv preprint arXiv:2507.04786 (2025).
 [57] Jiajun Huang, Sheng Di, Yafan Huang, Zizhong Chen, Franck Cappello, Yanfei Guo, and Rajeev Thakur. 2025. ghZCCL: Advancing GPU-aware
      Collective Communications with Homomorphic Compression. In Proceedings of the 39th ACM International Conference on Supercomputing (ICS ’25).
      Association for Computing Machinery, New York, NY, USA, 43–56. https://doi.org/10.1145/3721145.3733642
 [58] Jiajun Huang, Sheng Di, Xiaodong Yu, Yujia Zhai, Jinyang Liu, Yafan Huang, Ken Raffenetti, Hui Zhou, Kai Zhao, Xiaoyi Lu, Zizhong Chen, Franck
      Cappello, Yanfei Guo, and Rajeev Thakur. 2024. gZCCL: Compression-Accelerated Collective Communication Framework for GPU Clusters. In
      Proceedings of the 38th ACM International Conference on Supercomputing (Kyoto, Japan) (ICS ’24). Association for Computing Machinery, New York,
      NY, USA, 437–448. https://doi.org/10.1145/3650200.3656636
 [59] Jiajun Huang, Sheng Di, Xiaodong Yu, Yujia Zhai, Jinyang Liu, Zizhe Jian, Xin Liang, Kai Zhao, Xiaoyi Lu, Zizhong Chen, Franck Cappello,
      Yanfei Guo, and Rajeev Thakur. 2024. hZCCL: Accelerating Collective Communication with Co-Designed Homomorphic Compression. In SC24:
      International Conference for High Performance Computing, Networking, Storage and Analysis. 1–15. https://doi.org/10.1109/SC41406.2024.00110
 [60] Alexaner Ishii and Ryan Wells. 2023. The NVLink-Network Switch: NVIDIA’s Switch Chip for High Communication-Bandwidth Superpods.
      https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf.


Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                                                            33


[61] Ismayil Ismayilov, Javid Baydamirli, Doğan Sağbili, Mohamed Wahib, and Didem Unat. 2023. Multi-GPU Communication Schemes for Iterative
     Solvers: When CPUs Are Not in Charge. In Proceedings of the 37th International Conference on Supercomputing (Orlando, FL, USA) (ICS ’23).
     Association for Computing Machinery, New York, NY, USA, 192–202. https://doi.org/10.1145/3577193.3593713
[62] Mohammad Kefah Taha Issa, Muhammad Aditya Sasongko, Ilyas Turimbetov, Javid Baydamirli, Doğan Sağbili, and Didem Unat. 2024. Snoopie: A
     Multi-GPU Communication Profiler and Visualizer. In Proceedings of the 38th ACM International Conference on Supercomputing (Kyoto, Japan) (ICS
     ’24). Association for Computing Machinery, New York, NY, USA, 525–536. https://doi.org/10.1145/3650200.3656597
[63] John Jacobson, Martin Burtscher, and Ganesh Gopalakrishnan. 2024. HiRace: Accurate and Fast Source-Level Race Checking of GPU Programs.
     arXiv:2401.04701 [cs.DC]
[64] Ziyang Jia, Laxmi N. Bhuyan, and Daniel Wong. 2024. PCCL: Energy-Efficient LLM Training with Power-Aware Collective Communication. In 2024
     IEEE 42nd International Conference on Computer Design (ICCD). 84–91. https://doi.org/10.1109/ICCD63220.2024.00023
[65] Benjamin Klenk, Lena Oden, and Holger Froening. 2014. Analyzing Put/Get APIs for Thread-Collaborative Processors. In 2014 43rd International
     Conference on Parallel Processing Workshops. 411–418. https://doi.org/10.1109/ICPPW.2014.61
[66] Benjamin Klenk, Lena Oden, and Holger Fröning. 2014. GPU-centric communication for improved efficiency. In International Workshop on Green
     Programming, Computing and Data Processing (GPCDP) in conjunction with International Green Computing Conference (IGCC), Dallas, TX, USA.
[67] Benjamin Klenk, Lena Oden, and Holger Froning. 2015. Analyzing communication models for distributed thread-collaborative processors in
     terms of energy and time. In 2015 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). 318–327. https:
     //doi.org/10.1109/ISPASS.2015.7095817
[68] JaeHyuk Kwack, Colleen Bertoni, Umesh Unnikrishnan, Riccardo Balin, Khalid Hossain, Yasaman Ghadar, Timothy J. Williams, Abhishek Bagusetty,
     Mathialakan Thavappiragasam, Väinö Hatanpää, Archit Vasan, John Tramm, and Scott Parker. 2025. AI and HPC Applications on Leadership
     Computing Platforms: Performance and Scalability Studies. In 2025 IEEE International Parallel and Distributed Processing Symposium (IPDPS).
     210–222. https://doi.org/10.1109/IPDPS64566.2025.00027
[69] HPDPS Lab. 2025. COCCL: Compression and precision cO-awareness Collective Communication Library implemented based on NCCL. https:
     //github.com/hpdps-group/coccl. Accessed: 2025-11-02.
[70] Hicrest Laboratory. 2025. Blink GPU Benchmark. https://github.com/HicrestLaboratory/Blink-GPU.
[71] Akhil Langer and Jim Dinan. 2021. NVSHMEM: GPU-Integrated Communication for NVIDIA GPU Clusters. https://www.nvidia.com/en-us/on-
     demand/session/gtcspring21-s32515/.
[72] Lattice. 2023. QUDA Repository. https://github.com/lattice/quda.
[73] Michael LeBeane, Khaled Hamidouche, Brad Benton, Mauricio Breternitz, Steven K. Reinhardt, and Lizy K. John. 2017. GPU Triggered Networking
     for Intra-Kernel Communications (SC ’17). Association for Computing Machinery, New York, NY, USA, Article 22, 12 pages. https://doi.org/10.
     1145/3126908.3126950
[74] Michael LeBeane, Khaled Hamidouche, Brad Benton, Mauricio Breternitz, Steven K. Reinhardt, and Lizy K. John. 2018. ComP-Net: Command
     Processor Networking for Efficient Intra-Kernel Communications on GPUs. In Proceedings of the 27th International Conference on Parallel Architectures
     and Compilation Techniques (Limassol, Cyprus) (PACT ’18). Association for Computing Machinery, New York, NY, USA, Article 29, 13 pages.
     https://doi.org/10.1145/3243176.3243179
[75] Ang Li, Shuaiwen Leon Song, Jieyang Chen, Jiajia Li, Xu Liu, Nathan R. Tallent, and Kevin J. Barker. 2020. Evaluating Modern GPU Interconnect: PCIe,
     NVLink, NV-SLI, NVSwitch and GPUDirect. IEEE Trans. Parallel Distrib. Syst. 31, 1 (jan 2020), 94–110. https://doi.org/10.1109/TPDS.2019.2928289
[76] Ziming Li, Chenyang Hei, Fuliang Li, Tongrui Liu, Chengxi Gao, Xiuzhu Sha, and Xingwei Wang. 2025. TuCCL: Tailored and Unified Configuration
     Optimizations for High-Performance Collective Communication Library. In 2025 IEEE 33rd International Conference on Network Protocols (ICNP).
     1–11. https://doi.org/10.1109/ICNP65844.2025.11192370
[77] K. V. Manian, A. A. Ammar, A. Ruhela, C.-H. Chu, H. Subramoni, and D. K. Panda. 2019. Characterizing CUDA Unified Memory (UM)-Aware
     MPI Designs on Modern GPU Architectures. In Proceedings of the 12th Workshop on General Purpose Processing Using GPUs (Providence, RI, USA)
     (GPGPU ’19). Association for Computing Machinery, New York, NY, USA, 43–52. https://doi.org/10.1145/3300053.3319419
[78] Naoya Maruyama, Brian Van Essen, Jan Ciesko, Jeremiah Wilke, Christian Trott, Chung-Hsing Hsu, Neena Imam, Jim Dinan, Akhil Langer,
     CJ Newburn, and Sreeram Potluri. 2020. Scaling Scientific Computing with NVSHMEM. https://developer.nvidia.com/blog/scaling-scientific-
     computing-with-nvshmem/.
[79] Hans Meuer, Erich Strohmaier, Jack Dongarra, Horst Simon, and Martin Meuer. 2023. TOP500 Supercomputer Sites. https://www.top500.org/
[80] Takefumi Miyoshi, Hidetsugu Irie, Keigo Shima, Hiroki Honda, Masaaki Kondo, and Tsutomu Yoshinaga. 2012. FLAT: A GPU Programming
     Framework to Provide Embedded MPI. In Proceedings of the 5th Annual Workshop on General Purpose Processing with Graphics Processing Units
     (London, United Kingdom) (GPGPU-5). Association for Computing Machinery, New York, NY, USA, 20–29. https://doi.org/10.1145/2159430.2159433
[81] Timothy Prickett Morgan. 2024. Key Hyperscalers And Chip Makers Gang Up On Nvidia’s NVSwitch Interconnect. https://www.hpcwire.com/
     2024/05/30/everyone-except-nvidia-forms-ultra-accelerator-link-ualink-consortium/.
[82] Yasser Morsy and Alan George. 2025. Comparative Analysis of Parallel Communication Models for Regular and Irregular GPU Workloads. In
     Practice and Experience in Advanced Research Computing 2025: The Power of Collaboration (PEARC ’25). Association for Computing Machinery, New
     York, NY, USA, Article 34, 4 pages. https://doi.org/10.1145/3708035.3736073
[83] Harini Muthukrishnan, David Nellans, Daniel Lustig, Jeffrey A. Fessler, and Thomas F. Wenisch. 2021. Efficient Multi-GPU Shared Memory via
     Automatic Optimization of Fine-Grained Transfers. In 2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA).
                                                                                                                          Manuscript submitted to ACM
34                                                                                                                                  D. Unat et al.


      139–152. https://doi.org/10.1109/ISCA52012.2021.00020
 [84] Naveen Namashivayam, Krishna Kandalla, James B White III au2, Larry Kaplan, and Mark Pagel. 2023. Exploring Fully Offloaded GPU Stream-Aware
      Message Passing. arXiv:2306.15773 [cs.DC]
 [85] Naveen Namashivayam, Krishna Kandalla, Trey White, Nick Radcliffe, Larry Kaplan, and Mark Pagel. 2022. Exploring GPU Stream-Aware Message
      Passing using Triggered Operations. arXiv:2208.04817 [cs.DC]
 [86] NVIDIA. [n. d.]. https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#racecheck-tool
 [87] NVIDIA. 2011. CUDA 4.0 Release Notes. https://developer.nvidia.com/cuda-toolkit-40.
 [88] NVIDIA. 2012. NVIDIA GPUDirect™ Technology. https://developer.download.nvidia.com/devzone/devcenter/cuda/docs/GPUDirect_Technology_
      Overview.pdf.
 [89] NVIDIA. 2016. Fast Multi-GPU collectives with NCCL. https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/.
 [90] NVIDIA. 2017. CUDA 4.1 Release Notes. https://developer.nvidia.com/cuda-toolkit-41-archive.
 [91] NVIDIA. 2017. NVIDIA DGX-1 With Tesla V100 System Architecture. https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-
      whitepaper.pdf.
 [92] NVIDIA. 2021. Improving GPU Memory Oversubscription Performance. https://developer.nvidia.com/blog/improving-gpu-memory-
      oversubscription-performance/.
 [93] NVIDIA. 2023. CUDA Programming Guide Release 12.2. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html.
 [94] NVIDIA. 2023. CUDA Runtime - Device Management. https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group_
      _CUDART__DEVICE.
 [95] NVIDIA. 2023. DGX-2. https://www.nvidia.com/en-gb/data-center/dgx-2/.
 [96] NVIDIA. 2023. GPUDirect RDMA. https://docs.nvidia.com/cuda/gpudirect-rdma/.
 [97] NVIDIA. 2023. Magnum IO GDRCopy. https://developer.nvidia.com/gdrcopy.
 [98] NVIDIA. 2023. NVIDIA GPUDirect Family. https://developer.nvidia.com/gpudirect.
 [99] NVIDIA. 2023. NVLink and NVSwitch. https://www.nvidia.com/en-us/data-center/nvlink/.
[100] NVIDIA. 2023. NVSHMEM. https://developer.nvidia.com/nvshmem.
[101] NVIDIA. 2023. NVSHMEM 2.7.0 Release Notes. https://docs.nvidia.com/nvshmem/release-notes/release-270.html#release-270.
[102] NVIDIA. 2024. NVIDIA GB200. https://www.nvidia.com/en-us/data-center/gb200-nvl72/.
[103] NVIDIA. 2024. NVIDIA Grace Hopper Superchip. https://resources.nvidia.com/en-us-grace-cpu/nvidia-grace-hopper.
[104] NVIDIA. 2024. NVIDIA NVLink SGXLS10 Switch Systems User Manual. https://docs.nvidia.com/networking/display/sgxh100/introduction.
[105] NVIDIA. 2025. NCCL Tests. https://github.com/NVIDIA/nccl-tests.
[106] NVIDIA. November 2025. Technical Blog on NCCL 2.28 Release. https://developer.nvidia.com/blog/fusing-communication-and-compute-with-
      new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/.
[107] Lena Oden and Holger Fröning. 2013. GGAS: Global GPU address spaces for efficient communication in heterogeneous clusters. 2013 IEEE
      International Conference on Cluster Computing (CLUSTER) (2013), 1–8.
[108] Lena Oden, Holger Fröning, and Franz-Joseph Pfreundt. 2014. Infiniband-Verbs on GPU: A Case Study of Controlling an Infiniband Network Device
      from the GPU. In 2014 IEEE International Parallel & Distributed Processing Symposium Workshops. 976–983. https://doi.org/10.1109/IPDPSW.2014.111
[109] Lena Oden, Benjamin Klenk, and Holger Fröning. 2014. Energy-Efficient Collective Reduce and Allreduce Operations on Distributed GPUs. In 2014
      14th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing. 483–492. https://doi.org/10.1109/CCGrid.2014.21
[110] Lena Oden, Benjamin Klenk, and Holger Fröning. 2014. Energy-Efficient Stencil Computations on Distributed GPUs Using Dynamic Parallelism
      and GPU-Controlled Communication. In 2014 Energy Efficient Supercomputing Workshop. 31–40. https://doi.org/10.1109/E2SC.2014.14
[111] OpenMPI. 2023. Open MPI v5.0.x Documentation: CUDA. https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/cuda.html.
[112] OpenMPI. 2023. Open MPI v5.0.x Documentation: ROCm. https://docs.open-mpi.org/en/v5.0.x/tuning-apps/networking/rocm.html.
[113] Sreeram Potluri Pak Markthub, Jim Dinan and Seth Howell. 2022. Improving Network Performance of HPC Systems Using NVIDIA Magnum IO
      NVSHMEM and GPUDirect Async. https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-
      io-nvshmem-and-gpudirect-async/.
[114] Carl Pearson. 2023. Interconnect Bandwidth Heterogeneity on AMD MI250x and Infinity Fabric. arXiv:2302.14827 [cs.DC]
[115] Carl Pearson, Abdul Dakkak, Sarah Hashash, Cheng Li, I-Hsin Chung, Jinjun Xiong, and Wen-Mei Hwu. 2019. Evaluating Characteristics of
      CUDA Communication Primitives on High-Bandwidth Interconnects. In Proceedings of the 2019 ACM/SPEC International Conference on Performance
      Engineering (Mumbai, India) (ICPE ’19). Association for Computing Machinery, New York, NY, USA, 209–218. https://doi.org/10.1145/3297663.
      3310299
[116] Sreeram Potluri, Anshuman Goswami, Davide Rossetti, C.J. Newburn, Manjunath Gorentla Venkata, and Neena Imam. 2017. GPU-Centric
      Communication on NVIDIA GPU Clusters with InfiniBand: A Case Study with OpenSHMEM. In 2017 IEEE 24th International Conference on High
      Performance Computing (HiPC). 253–262. https://doi.org/10.1109/HiPC.2017.00037
[117] Sreeram Potluri, Anshuman Goswami, Manjunath Gorentla Venkata, and Neena Imam. 2018. Efficient Breadth First Search on Multi-GPU Systems
      Using GPU-Centric OpenSHMEM. In OpenSHMEM and Related Technologies. Big Compute and Big Data Convergence, Manjunath Gorentla Venkata,
      Neena Imam, and Swaroop Pophale (Eds.). Springer International Publishing, Cham, 82–96.


Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                                                        35


[118] Sreeram Potluri, Anshuman Goswami, Manjunath Gorentla Venkata, and Neena Imam. 2018. Efficient Breadth First Search on Multi-GPU Systems
      Using GPU-Centric OpenSHMEM. In OpenSHMEM and Related Technologies. Big Compute and Big Data Convergence, Manjunath Gorentla Venkata,
      Neena Imam, and Swaroop Pophale (Eds.). Springer International Publishing, Cham, 82–96.
[119] Sreeram Potluri, Khaled Hamidouche, Akshay Venkatesh, Devendar Bureddy, and Dhabaleswar K. Panda. 2013. Efficient Inter-node MPI
      Communication Using GPUDirect RDMA for InfiniBand Clusters with NVIDIA GPUs. In 2013 42nd International Conference on Parallel Processing.
      80–89. https://doi.org/10.1109/ICPP.2013.17
[120] Sreeram Potluri, Davide Rossetti, Donald Becker, Duncan Poole, Manjunath Gorentla Venkata, Oscar Hernandez, Pavel Shamis, M. Graham Lopez,
      Mathew Baker, and Wendy Poole. 2015. Exploring OpenSHMEM Model to Program GPU-Based Extreme-Scale Systems. In Revised Selected Papers
      of the Second Workshop on OpenSHMEM and Related Technologies. Experiences, Implementations, and Technologies - Volume 9397 (Annapolis, MD,
      USA) (OpenSHMEM 2015). Springer-Verlag, Berlin, Heidelberg, 18–35. https://doi.org/10.1007/978-3-319-26428-8_2
[121] S. Potluri, H. Wang, D. Bureddy, A.K. Singh, C. Rosales, and Dhabaleswar K. Panda. 2012. Optimizing MPI Communication on Multi-GPU Systems
      Using CUDA Inter-Process Communication. In 2012 IEEE 26th International Parallel and Distributed Processing Symposium Workshops & PhD Forum.
      1848–1857. https://doi.org/10.1109/IPDPSW.2012.228
[122] Kishore Punniyamurthy, Bradford M. Beckmann, and Khaled Hamidouche. 2023. GPU-initiated Fine-grained Overlap of Collective Communication
      with Computation. arXiv:2305.06942 [cs.DC]
[123] Kishore Punniyamurthy, Khaled Hamidouche, and Bradford M Beckmann. 2023. Optimizing Distributed ML Communication with Fused
      Computation-Collective Operations. arXiv:2305.06942
[124] Zaid Qureshi, Vikram Sharma Mailthody, Isaac Gelado, Seungwon Min, Amna Masood, Jeongmin Park, Jinjun Xiong, C. J. Newburn, Dmitri
      Vainbrand, I-Hsin Chung, Michael Garland, William Dally, and Wen-mei Hwu. 2023. GPU-Initiated On-Demand High-Throughput Storage Access
      in the BaM System Architecture. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages
      and Operating Systems, Volume 2 (Vancouver, BC, Canada) (ASPLOS 2023). Association for Computing Machinery, New York, NY, USA, 325–339.
      https://doi.org/10.1145/3575693.3575748
[125] Davide Rossetti, Sreeram Potluri, and David Fontaine. 2016. State of GPUDirect Technologies. https://on-demand.gputechconf.com/gtc/2016/
      presentation/s6264-davide-rossetti-GPUDirect.pdf.
[126] Doğan Sağbili, Sinan Ekmekçibaşı, Khaled Z. Ibrahim, Tan Nguyen, and Didem Unat. 2025. Uniconn: A Uniform High-Level Communication
      Library for Portable Multi-GPU Programming. In 2025 IEEE International Conference on Cluster Computing (CLUSTER). 1–12. https://doi.org/10.
      1109/CLUSTER59342.2025.11186498
[127] Gabin Schieffer, Ruimin Shi, Stefano Markidis, Andreas Herten, Jennifer Faj, and Ivy Peng. 2025. Understanding Data Movement in AMD Multi-GPU
      Systems with Infinity Fabric. In Proceedings of the SC ’24 Workshops of the International Conference on High Performance Computing, Network,
      Storage, and Analysis (Atlanta, GA, USA) (SC-W ’24). IEEE Press, 567–576. https://doi.org/10.1109/SCW63240.2024.00079
[128] Tim C. Schroeder. 2011. Peer-to-Peer & Unified Virtual Addressing. https://developer.download.nvidia.com/CUDA/training/cuda_webinars_
      GPUDirect_uva.pdf.
[129] Kawthar Shafie Khorassani, Jahanzeb Hashmi, Ching-Hsiang Chu, Chen-Chun Chen, Hari Subramoni, and Dhabaleswar K. Panda. 2021. Designing
      a ROCm-Aware MPI Library for AMD GPUs: Early Experiences. In High Performance Computing, Bradford L. Chamberlain, Ana-Lucia Varbanescu,
      Hatem Ltaief, and Piotr Luszczek (Eds.). Springer International Publishing, Cham, 118–136.
[130] Aashaka Shah, Vijay Chidambaram, Meghan Cowan, Saeed Maleki, Madan Musuvathi, Todd Mytkowicz, Jacob Nelson, Olli Saarikivi, and Rachee
      Singh. 2022. TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches. arXiv:2111.04867 [cs.DC]
[131] Aashaka Shah, Abhinav Jangda, Binyang Li, Caio Rocha, Changho Hwang, Jithin Jose, Madan Musuvathi, Olli Saarikivi, Peng Cheng, Qinghua
      Zhou, Roshan Dathathri, Saeed Maleki, and Ziyue Yang. 2025. MSCCL++: Rethinking GPU Communication Abstractions for Cutting-edge AI
      Applications. https://doi.org/10.48550/arXiv.2504.09014
[132] Gilad Shainer, Ali Ayoub, Pak Lui, Tong Liu, Michael Kagan, Christian R. Trott, Greg Scantlen, and Paul S. Crozier. 2011. The development of
      Mellanox/NVIDIA GPUDirect over InfiniBand—a new model for GPU to GPU communications. Computer Science - Research and Development 26, 3
      (01 Jun 2011), 267–273. https://doi.org/10.1007/s00450-011-0157-1
[133] Pavel Shamis, Manjunath Gorentla Venkata, M Graham Lopez, Matthew B Baker, Oscar Hernandez, Yossi Itigin, Mike Dubman, Gilad Shainer,
      Richard L Graham, Liran Liss, et al. 2015. UCX: an open source framework for HPC network APIs and beyond. In 2015 IEEE 23rd Annual Symposium
      on High-Performance Interconnects. IEEE, 40–43.
[134] Chuanming Shao, Jinyang Guo, Pengyu Wang, Jing Wang, Chao Li, and Minyi Guo. 2022. Oversubscribing GPU Unified Virtual Memory:
      Implications and Suggestions. In Proceedings of the 2022 ACM/SPEC on International Conference on Performance Engineering (Beijing, China) (ICPE
      ’22). Association for Computing Machinery, New York, NY, USA, 67–75. https://doi.org/10.1145/3489525.3511691
[135] Rong Shi, Sreeram Potluri, Khaled Hamidouche, Jonathan Perkins, Mingzhe Li, Davide Rossetti, and Dhabaleswar K. D K Panda. 2014. Designing
      efficient small message transfer mechanism for inter-node MPI communication on InfiniBand GPU clusters. In 2014 21st International Conference on
      High Performance Computing (HiPC). 1–10. https://doi.org/10.1109/HiPC.2014.7116873
[136] Takashi Shimokawabe, Takayuki Aoki, Tomohiro Takaki, Toshio Endo, Akinori Yamanaka, Naoya Maruyama, Akira Nukada, and Satoshi Matsuoka.
      2011. Peta-Scale Phase-Field Simulation for Dendritic Solidification on the TSUBAME 2.0 Supercomputer. In Proceedings of 2011 International
      Conference for High Performance Computing, Networking, Storage and Analysis (Seattle, Washington) (SC ’11). Association for Computing Machinery,
      New York, NY, USA, Article 3, 11 pages. https://doi.org/10.1145/2063384.2063388
                                                                                                                       Manuscript submitted to ACM
36                                                                                                                                   D. Unat et al.


[137] Min Si, Yongzhou Chen Pavan Balaji, Ching-Hsiang Chu, Adi Gangidi, Saif Hasan, Subodh Iyengar, Dan Johnson, Bingzhe Liu, Regina Ren,
      Ashmitha Jeevaraj Shetty, Greg Steinbrecher, Yulun Wang, Bruce Wu, Xinfeng Xie, Jingyi Yang, Mingran Yang, Kenny Yu, Minlan Yu, Cen Zhao,
      Wes Bland, Denis Boyda, Suman Gumudavelli, Prashanth Kannan, Cristian Lumezanu, Rui Miao, Zhe Qu, Venkat Ramesh, Maxim Samoylov, Jan
      Seidel, Srikanth Sundaresan, Feng Tian, Qiye Tan, Shuqiang Zhang, Yimeng Zhao, Shengbao Zheng, Art Zhu, and Hongyi Zeng. 2025. Collective
      Communication for 100k+ GPUs. https://doi.org/10.48550/arXiv.2510.20171
[138] Mark Silberstein, Bryan Ford, Idit Keidar, and Emmett Witchel. 2014. GPUfs: Integrating a File System with GPUs. ACM Trans. Comput. Syst. 32, 1,
      Article 1 (feb 2014), 31 pages. https://doi.org/10.1145/2553081
[139] Mark Silberstein, Sangman Kim, Seonggu Huh, Xinya Zhang, Yige Hu, Amir Wated, and Emmett Witchel. 2016. GPUnet: Networking Abstractions
      for GPU Programs. 34, 3, Article 9 (sep 2016), 31 pages. https://doi.org/10.1145/2963098
[140] Siddharth Singh, Mahua Singh, and Abhinav Bhatele. 2025. The Big Send-off: High Performance Collectives on GPU-based Supercomputers.
      arXiv:2504.18658 [cs.DC] https://arxiv.org/abs/2504.18658
[141] Benjamin Spector, Jordan Juravsky, Stuart Sul, Dylan Lim, Owen Dugan, Simran Arora, and Chris Ré. 2025. We Bought the Whole GPU, So We’re
      Damn Well Going to Use the Whole GPU. https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main.
[142] Jeff A. Stuart, Michael Cox, and John D. Owens. 2010. GPU-to-CPU Callbacks. In Proceedings of the 2010 Conference on Parallel Processing (Ischia,
      Italy) (Euro-Par 2010). Springer-Verlag, Berlin, Heidelberg, 365–372.
[143] Stuart H. Sul, Simran Arora, Benjamin F. Spector, and Christopher Ré. 2025. ParallelKittens: Systematic and Practical Simplification of Multi-GPU
      AI Kernels. arXiv:2511.13940 [cs.DC] https://arxiv.org/abs/2511.13940
[144] The Unified Communication X Library [n. d.]. The Unified Communication X Library. http://www.openucx.org.
[145] Adam Thompson and Newburn C.J. 2012. GPUDirect Storage: A Direct Path Between Storage and GPU Memory. https://developer.nvidia.com/
      blog/gpudirect-storage/.
[146] Christian Trott. 2018. Early Experience with NVSHMEM: Extending the Kokkos Programming Model with PGAS Semantics. https://www.osti.gov/
      servlets/purl/1806950.
[147] James D. Trotter, Sinan Ekmekçibaşı, Doğan Sağbili, Johannes Langguth, Xing Cai, and Didem Unat. 2025. CPU- and GPU-initiated Communication
      Strategies for Conjugate Gradient Methods on Large GPU Clusters. In Proceedings of the International Conference for High Performance Computing,
      Networking, Storage and Analysis (SC ’25). Association for Computing Machinery, New York, NY, USA, 298–315. https://doi.org/10.1145/3712285.
      3759774
[148] Ilyas Turimbetov, Mohamed Wahib, and Didem Unat. 2025. A Device-Side Execution Model for Multi-GPU Task Graphs. In Proceedings of
      the 39th ACM International Conference on Supercomputing (ICS ’25). Association for Computing Machinery, New York, NY, USA, 384–396.
      https://doi.org/10.1145/3721145.3730426
[149] Didem Unat, Anshu Dubey, Emmanuel Jeannot, and John Shalf. 2025. The Persistent Challenge of Data locality in Post-Exascale Era. Computing in
      Science and Engineering (2025), 1–9. https://doi.org/10.1109/MCSE.2025.3567586
[150] Didem Unat, Ilyas Turimbetov, Mohammad Kefah Taha Issa, Dogan Sagbili, Flavio Vella, Daniele De Sensi, and Ismayil Ismayilov. 2026. Figures of
      The Landscape of GPU-Centric Communication. https://doi.org/10.6084/m9.figshare.32032020. Accessed: 2026-04-16.
[151] Manjunath Gorentla Venkata, Valentine Petrov, Sergey Lebedev, Devendar Bureddy, Ferrol Aderholdt, Joshua Ladd, Gil Bloch, Mike Dubman, and
      Gilad Shainer. 2025. Unified Collective Communication: A Unified Library for CPU, GPU, and DPU Collectives . IEEE Micro 45, 02 (March 2025),
      26–35. https://doi.org/10.1109/MM.2025.3534638
[152] Akshay Venkatesh, Khaled Hamidouche, Sreeram Potluri, Davide Rosetti, Ching-Hsiang Chu, and Dhabaleswar K. Panda. 2017. MPI-GDS: High
      Performance MPI Designs with GPUDirect-aSync for CPU-GPU Control Flow Decoupling. In 2017 46th International Conference on Parallel
      Processing (ICPP). 151–160. https://doi.org/10.1109/ICPP.2017.24
[153] Ján Veselý, Arkaprava Basu, Abhishek Bhattacharjee, Gabriel H. Loh, Mark Oskin, and Steven K. Reinhardt. 2018. Generic System Calls for GPUs.
      In 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA). 843–856. https://doi.org/10.1109/ISCA.2018.00075
[154] Mathias Wagner. 2020. GTC 20: Overcoming Latency Barriers: Strong Scaling HPC Applications with NVSHMEM. https://www.nvidia.com/en-
      us/on-demand/session/gtcsj20-s21673/.
[155] Guanhua Wang, Shivaram Venkataraman, Amar Phanishayee, Jorgen Thelin, Nikhil Devanur, and Ion Stoica. 2019. Blink: Fast and Generic
      Collectives for Distributed ML. arXiv:1910.04940 [cs.DC]
[156] Hao Wang, Sreeram Potluri, Devendar Bureddy, Carlos Rosales, and Dhabaleswar K. Panda. 2014. GPU-Aware MPI on RDMA-Enabled Clusters:
      Design, Implementation and Evaluation. IEEE Transactions on Parallel and Distributed Systems 25, 10 (2014), 2595–2605. https://doi.org/10.1109/
      TPDS.2013.222
[157] Hao Wang, Sreeram Potluri, Miao Luo, Ashish Singh, Sayantan Sur, and D.K. Panda. 2011. MVAPICH2GPU: optimized GPU to GPU communication
      for InfiniBand clusters. Computer Science - Research and Development 26 (06 2011), 257–266. https://doi.org/10.1007/s00450-011-0171-3
[158] Hao Wang, Sreeram Potluri, Miao Luo, Ashish Kumar Singh, Xiangyong Ouyang, Sayantan Sur, and Dhabaleswar K. Panda. 2011. Optimized Non-
      contiguous MPI Datatype Communication for GPU Clusters: Design, Implementation and Evaluation with MVAPICH2. In 2011 IEEE International
      Conference on Cluster Computing. 308–316. https://doi.org/10.1109/CLUSTER.2011.42
[159] Yuke Wang, Boyuan Feng, Zheng Wang, Tong Geng, Ang Li, Kevin Barker, and Yufei Ding. 2023. MGG: Accelerating Graph Neural Networks with
      Fine-grained intra-kernel Communication-Computation Pipelining on Multi-GPU Platforms. In USENIX Symposium on Operating Systems Design
      and Implementation (OSDI’21).
Manuscript submitted to ACM
The Landscape of GPU-Centric Communication                                                                                                        37


[160] Adam Weingram, Yuke Li, Hao Qi, Darren Ng, Liuyao Dai, and Xiaoyi Lu. 2023. xCCL: A Survey of Industry-Led Collective Communication
      Libraries for Deep Learning. Journal of Computer Science and Technology 38, 1 (01 Feb 2023), 166–195. https://doi.org/10.1007/s11390-023-2894-6
[161] Wei Wu, George Bosilca, Rolf vandeVaart, Sylvain Jeaugey, and Jack Dongarra. 2016. GPU-Aware Non-Contiguous Data Movement In Open
      MPI. In Proceedings of the 25th ACM International Symposium on High-Performance Parallel and Distributed Computing (Kyoto, Japan) (HPDC ’16).
      Association for Computing Machinery, New York, NY, USA, 231–242. https://doi.org/10.1145/2907294.2907317
[162] CHENHAO XIE, Jieyang Chen, Jesun Firoz, Jiajia Li, Shuaiwen Leon Song, Kevin Barker, Mark Raugas, and Ang Li. 2021. Fast and Scalable Sparse
      Triangular Solver for Multi-GPU Based HPC Architectures. In Proceedings of the 50th International Conference on Parallel Processing (Lemont, IL,
      USA) (ICPP ’21). Association for Computing Machinery, New York, NY, USA, Article 53, 11 pages. https://doi.org/10.1145/3472456.3472478
[163] Guanbin Xu, Zhihao Le, Yinhe Chen, Zhiqi Lin, Zewen Jin, Youshan Miao, and Cheng Li. 2025. AutoCCL: Automated Collective Communication
      Tuning for Accelerating Distributed and Parallel DNN Training. In 22nd USENIX Symposium on Networked Systems Design and Implementation
      (NSDI 25). USENIX Association, Philadelphia, PA, 667–683. https://www.usenix.org/conference/nsdi25/presentation/xu-guanbin
[164] Junchao Zhang, Jed Brown, Satish Balay, Jacob Faibussowitsch, Matthew Knepley, Oana Marin, Richard Tran Mills, Todd Munson, Barry F. Smith,
      and Stefano Zampini. 2021. The PetscSF Scalable Communication Layer. arXiv:2102.13018 [cs.DC]
[165] Lingqi Zhang, Mohamed Wahib, Peng Chen, Jintao Meng, Xiao Wang, Toshio Endo, and Satoshi Matsuoka. 2023. PERKS: A Locality-Optimized
      Execution Model for Iterative Memory-Bound GPU Applications. In Proceedings of the 37th International Conference on Supercomputing (Orlando,
      FL, USA) (ICS ’23). Association for Computing Machinery, New York, NY, USA, 167–179. https://doi.org/10.1145/3577193.3593705
[166] Hui Zhou, Ken Raffenetti, Yanfei Guo, and Rajeev Thakur. 2022. MPIX Stream: An Explicit Solution to Hybrid MPI+X Programming. In Proceedings
      of the 29th European MPI Users’ Group Meeting (Chattanooga, TN, USA) (EuroMPI/USA ’22). Association for Computing Machinery, New York, NY,
      USA, 1–10. https://doi.org/10.1145/3555819.3555820
[167] Qinghua Zhou, Pouya Kousha, Quentin Anthony, Kawthar Shafie Khorassani, Aamir Shafi, Hari Subramoni, and Dhabaleswar K. Panda. 2022.
      Accelerating MPI All-to-All Communication with Online Compression on Modern GPU Clusters. In High Performance Computing: 37th International
      Conference, ISC High Performance 2022, Hamburg, Germany, May 29 – June 2, 2022, Proceedings (Hamburg, Germany). Springer-Verlag, Berlin,
      Heidelberg, 3–25. https://doi.org/10.1007/978-3-031-07312-0_1




                                                                                                                       Manuscript submitted to ACM
