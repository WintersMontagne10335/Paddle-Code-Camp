# 一、 ImplicitGemm

- [一、ImplicitGemm](#一-ImplicitGemm)
  - [0. 背景知识](#0-背景知识)

## 0. 背景知识

访问全局存储（Global Memory）时，同一 Warp 中的相邻线程访问连续的地址，访存请求会被合并，合并的访存能够最大化 Global Memory 的吞吐。

访问 Global Memory 时，尽可能使用最宽的数据类型（float4）进行访问，这样可以最大化访存指令的利用率。

CUDA 的共享存储（Shared Memory）按照每 4Bytes 划分为一个 bank，共分为 32 个 bank。当同一 Warp 中的线程访问同一 bank 的不同地址时会发生冲突（bank conflict）。
无 bank conflict 的访存模式才能最大化 Shared Memory 的吞吐。

TODO
