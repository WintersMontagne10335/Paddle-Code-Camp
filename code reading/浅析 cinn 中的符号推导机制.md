## 什么是符号推导
### 感性认识
![image](https://github.com/WintersMontagne10335/Paddle-Code-Camp/assets/118546135/1c5926c3-2990-49c4-a1dd-63b43a941dca)
![image](https://github.com/WintersMontagne10335/Paddle-Code-Camp/assets/118546135/662989da-1f7a-445b-b5ee-3e89d9c0dd69)

输⼊ Tensor 的**动态 shape 信息** -> 输出 Tensor 的 shape 信息。

> 例如 exp 算⼦的输⼊ X shape 为 [S2, 1] 时，则输出 Out 的 shape 同样也为 [S2, 1]。

### 流程概览
![9a1976fdf332b3e6cf007e636f39f1ab](https://github.com/WintersMontagne10335/Paddle-Code-Camp/assets/118546135/f5c6da29-c984-4e8e-a7f2-8a9c672361c9)

- 符号
- ShapeOrData
- Constraints

### 与 InferMeta 有什么区别与联系

InferMeta 有 shape 推导的功能。

为什么 cinn 不直接复用 InferMeta，而是另外开发了一套 shape 推导的工具。

|对比方向|InferMeta|InferSymbolicShape|
|---|---|---|
|未能推导出的维度大小|-1|新的符号 SX|
|约束信息|未保存|保存|
|应用场景|with_cinn=false 或 with_cinn=true 但是 tensor shape 为静态时|with_cinn=true 且 tensor shape 为静态时|

## 为什么需要符号推导
### 动态 shape 带来的挑战
- 编译器需要 shape 信息做优化
- 为每一个实际输入shape组合生成一份编译结果
- 动态 shape，且可能变化范围非常大

![image](https://github.com/WintersMontagne10335/Paddle-Code-Camp/assets/118546135/ba963a4d-2fbd-4195-9cea-631bbc6dc517)

### 常见的解决方法
Padding 成 static shape。
|方案|说明|分析|
|---|---|---|
|Naive padding|动态维度，padding 到最大长度|计算效率非常低|
|分桶|划分多个范围（64, 128，512, 2048, ...），选最小可用的模型，padding 到对应长度|最常用；可能产生组合爆炸|
|动态编译|第一次执行时，不优化。执行后，根据真实 shape 编译出高性能版本。下次遇到符合的 shape，使用高性能版本|实现复杂；不能保存太多的 compile cache|
|crop 数据|特别长的 input，数据截断不处理|会影响精度|

Shape Constraint。
虽然不知道 shape 具体的值，但是可以缩小范围。
> A + B = C 场景，Add 算子的输入 vector A 和 vector B，shape 是相同的

其它的方法。
- 改代码、换算子
- Control Flow 的支持
- Sequence Mask
- ...

### cinn 的解决方法
符号。
> 众所周知，⼀个算⼦的计算，在 shape 不同时，性能优化⽅式不同，⽐如需要不同的 Tile 形式。⽽在动态 shape 场景下，编译期⽆法确切
获得某些维度的shape值，这些shape由符号表示，这些符号的特性是：我们不知道其具体的值，但是在整个编译期间，其值不会发⽣改
变。

符号推导。
> 基础算子符号推导：手动实现，可以参照 InferMeta。
> 
> 组合算子、反向算子：依据拆解规则拆分成基础算子处理。
> 
> 图符号推导：对于⼀个 Program，可以从输⼊ Tensor 开始，以拓扑序的⽅式对所有算⼦进⾏符号推导便可以得到 Program 中所有 Tensor 的
符号维度信息。

约束。
> ⽤于保存PIR Program中每个Tensor与对应符号维度信息的映射关系。
> 
> ⽤于保存PIR Program不同维度符号间的约束关系（如相等，可⼴播等约束）。

分桶优化。

![image](https://github.com/WintersMontagne10335/Paddle-Code-Camp/assets/118546135/7f27f307-46ba-4a5e-bf04-982dc15d2429)

在运⾏时，所有的 shape 都可获取，因此可以将真实的 shap e值带⼊符号表达式，其值将落于⼀个区间，即可选择对应的 Bucket，进⽽选
择最优的 Kernel 进⾏执⾏。

### tvm relax 的解决方法


### 其它的解决方法
- TVM - relax
- TensorRT - Padding
- DISC - 阿里 基于 MLIR
- Nimble - AWS 基于 TVM
- DietCode - AWS 基于 TVM

## 符号推导是如何实现的
### 基础算子符号推导
可参照的 PR: [Add InferSymbolicShape for pd_op.nonzero](https://github.com/PaddlePaddle/Paddle/pull/62987)。

以上面 PR 中的 nonzero 为例，接下来会简述一下新增基础符号算子的一般开发流程。

InferSymbolicShape 的主体逻辑会与 InferMeta 相一致，所以我们可以先读懂 InferMeta 的代码，然后参照它，去写 InferSymbolicShape 的代码。

我们可以先在 ops.yaml 文件里面找到 nonezero 的 InferMeta 的函数名。

```yaml
- op : nonzero
  args : (Tensor condition)
  output : Tensor(out)
  infer_meta :
    func : NonZeroInferMeta
  kernel :
    func : nonzero
    data_type: condition
```

可以看到 nonzero 对应的 InferMeta 为 NonZeroInferMeta。

我们在 Paddle 仓库搜索这个名字，从而找到 函数主体。

```C++
// NonZeroInferMeta 在 unary.cc 文件里

void NonZeroInferMeta(const MetaTensor& condition, MetaTensor* out) {
  auto rank = condition.dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      1UL,
      phi::errors::InvalidArgument(
          "Input(Condition) should have number of dimension at least 1"));
  out->set_dims(common::make_ddim({-1, rank}));
  out->set_dtype(DataType::INT64);
}
```

接着我们根据 NonZeroInferMeta 的主体逻辑，开发 nonzero 对应的 InferSymbolicShape。

一般需要修改四个文件：
- paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/unary_infer_sym.h
- paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/unary_infer_sym.cc
- paddle/phi/api/yaml/ops.yaml
- test/ir/pir/cinn/symbolic/test_infer_sym_shape_unary_op.py

先说 unary_infer_sym.h，这个是头文件，我们参照该文件中的其它算子，添加下面的代码即可。

```C++
OP_DECLARE_INFER_SYMBOLIC_SHAPE(Nonzero)
```

然后我们在 unary_infer_sym.cc 添加函数主体，主要包含以下内容。
- 从 shape_analysis 中获取输⼊ Tensor 的符号维度（调⽤ GetShapeOrDataForValue ）
- shape 验证（可选）
- 根据算⼦的处理逻辑计算输出 Tensor 的符号维度
- 将输出 Tensor 的符号维度存⼊ shape_analysis 中（调⽤ SetShapeOrDataForValue ）

```C++
bool NonzeroOpInferSymbolicShape(
    pir::Operation *op, pir::ShapeConstraintIRAnalysis *shape_analysis) {
  const auto &x_shape_or_data =
      shape_analysis->GetShapeOrDataForValue(op->operand_source(0));
  const auto &x_shape = x_shape_or_data.shape();
  int rank = x_shape.size();

  PADDLE_ENFORCE_GE(
      rank,
      1UL,
      phi::errors::InvalidArgument(
          "Input(x) should have number of dimension at least 1."));

  std::string sym_name = shape_analysis->GetNextSymName();
  std::vector<symbol::DimExpr> out_shape{symbol::DimExpr{sym_name},
                                         symbol::DimExpr{rank}};

  symbol::ShapeOrDataDimExprs shape_data{
      symbol::TensorShapeOrDataDimExprs(out_shape)};
  shape_analysis->SetShapeOrDataForValue(op->result(0), shape_data);
  return true;
}
```

多数情况下，实现起来都比较简单，但是也有可能出现问题，有些是没有找到对应的方法，有些是 cinn 还没有支持。

一般情况下，我们可以通过阅读相似算子的实现方式来找解决思路，比如 NonZeroInferMeta 中 out 的输出 shape 含有 -1（表示动态维度大小），
我们不知道如何处理，我们可以通过在 unary.cc（或其它含有 InferMeta 的文件），搜索 -1，看看其它含有 -1 的算子的 
InferSymbolicShape 是如何实现的，这里我们调用

```C++
  std::string sym_name = shape_analysis->GetNextSymName();
```

用一个新的符号表示该维度大小。

这里补充一下如何添加约束，例：我们可以使用以下代码添加 input_dim_expr_vector[i], label_dim_expr_vector[i] 相等的约束。

```C++
    shape_analysis->AddEqualCstr(input_dim_expr_vector[i],
                                 label_dim_expr_vector[i]);
```

更多的约束类型，可以 ctrl +左键点进 AddEqualCstr 查看。

具体如何使用可以参照已有的相关代码。

接下来我们说一下 paddle/phi/api/yaml/ops.yaml，一般情况下我们直接在 ops.yaml 中算子相关内容的后面添加

```yaml
  interfaces : paddle::dialect::InferSymbolicShapeInterface
```

即可。比如 nonzero 修改完后，为

```yaml
- op : nonzero
  args : (Tensor condition)
  output : Tensor(out)
  infer_meta :
    func : NonZeroInferMeta
  kernel :
    func : nonzero
    data_type: condition
  interfaces : paddle::dialect::InferSymbolicShapeInterface
```

但是，某些情况下，可能在 paddle/phi/api/yaml/ops.yaml 中找不到对应的算子，而是在 paddle/phi/api/yaml/legacy_ops.yaml 中找到该算子，
比如 distribute_fpn_proposals 算子就是这样。这里我们可以参照一下 https://github.com/PaddlePaddle/Paddle/pull/63947/files，修改
- paddle/fluid/pir/dialect/operator/ir/ops.yaml。

最后说一下 test_infer_sym_shape_unary_op.py，TODO


注意，以上的内容基本都是针对的 nonzero，所以涉及到的文件为 unary_infer_sym.h, unary_infer_sym.cc, test_infer_sym_shape_unary_op.py，
其它算子可能不在 unary 下，需要根据 InferMeta 所在的文件作调整。

这部分任务比较简单，如果有同学想要入门 ai 编译器的，可以从这里入手。

目前可以通过以下渠道，找到待开发的任务。

> [TODO](https://github.com/search?q=repo%3APaddlePaddle%2FPaddle+path%3A%2F%5Epaddle%5C%2Ffluid%5C%2Fpir%5C%2Fdialect%5C%2Foperator%5C%2Finterface%5C%2Finfer_symbolic_shape%5C%2F%2F+todo&type=code)
>
> [Unimplement](https://github.com/search?q=repo%3APaddlePaddle%2FPaddle+path%3A%2F%5Epaddle%5C%2Ffluid%5C%2Fpir%5C%2Fdialect%5C%2Foperator%5C%2Finterface%5C%2Finfer_symbolic_shape%5C%2F%2F+unimplement&type=code)
>
> 子图报错相关的，可以联系留杰老师，[相关网址](https://github.com/PaddlePaddle/Paddle/issues/62930)

## 参考资料
- [动态 shape 的挑战与解决现状](https://zhuanlan.zhihu.com/p/661889518)
- [AI编译优化--Dynamic Shape Compiler](https://zhuanlan.zhihu.com/p/305546437)
- [深度学习框架中的动态Shape问题](https://blog.csdn.net/qianqing13579/article/details/125660401)
- [Relax Shape Computation Design](https://github.com/tlc-pack/relax/wiki/Relax-Shape-Computation-Design)
- [TVM Relax如何支持dynamic shape](https://zhuanlan.zhihu.com/p/627449108)
- [如何评价 TVM 在 Relay 之后的新 IR Relax？](https://www.zhihu.com/question/522101384)
- [TVM分析和使用](https://zhuanlan.zhihu.com/p/690256525)
- [Nimble(TVM 动态shape解决思路)论文分析](https://zhuanlan.zhihu.com/p/354995641)
