# 一、 ComputeInline

- [一、ComputeInline](#一-ComputeInline)
  - [1. 背景知识](#1-背景知识)
  - [2. 主要流程](#2-主要流程)
  - [3. LeafBlockRemovalPlan](#3-LeafBlockRemovalPlan)
  - [4. ComputeInliner](#4-ComputeInliner)

## 1. 背景知识

### Ⅰ. 访问者（Visitor） 模式
访问者模式是一种行为型设计模式，它将算法与其所作用的对象分离开来，使得能够在不改变对象结构的前提下，对对象中的元素进行新的操作。
该模式的核心思想是，定义一个访问者对象，并将其传递给需要被访问的对象，在对象接受访问者的访问时，会调用访问者对象中的函数，在该函数中实现对象对于访问者的响应操作。

以下图为例。

![image](https://github.com/WintersMontagne10335/Paddle-Code-Camp/assets/118546135/83ade3bc-1e76-4a0d-a63c-504d00f7990a)

首先，对于上面每个地方，每个人都可能有不同的访问行为，有游客到济公殿会看看墙上济公的故事，有的游客可能就对济公殿很熟悉，
进去拜一下就出来了，有的到了断桥残雪感慨一句全是人头，也有人到了断桥残雪，就会会回味一下张祜的诗句。

其次，对于上面每个地方，对于不同的游客来说，不确定，所以，不管是谁，给他准备个入口进来总不会有问题。

结合上面分析，我们可以初步分析访问者模式是怎么一回事儿：在每个游客自己内部，把对每个景点想操作的行为定义出来，然后在每个景点，准备个接口，把游客迎进来，
最后呢，把这个游客迎进来之后呢，执行这个游客里面定义的，关于自己的行为。

实现方式上，一般为，父类(BaseVisitor)含有所有节点（景点）的 visit 函数，这个 visit 函数什么都不做，只是按 dfs 的顺序调用其它节点（景点）的 visit 函数。
子类 visitor 继承父类，重写某些节点的 visit 函数（比如张三是子类 visitor，他到了断桥残雪，就会会回味一下张祜的诗句）。在子类 visit 函数的结尾，
如果想继续以 dfs 的顺序遍历，就应该调用父类的 visit 函数，再 return；如果想直接返回，直接 return 即可。

此模式常用于 CST（具体语法树） 到 AST（抽象语法树） 的转换。

## 2. 主要流程

ComputeInline 效果示例如下。

```
// 原 ir：
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(B)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              B[i0, i1, i2] = (1 + A[i0, i1, i2])
            }
          }
        }
      }
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(C)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              C[i0, i1, i2] = (2 * B[i1, i0, i2])
            }
          }
        }
      }
    }
  }
// ComputeInline 之后：
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 32)
        {
          serial for (k, 0, 32)
          {
            ScheduleBlock(C)
            {
              i0, i1, i2 = axis.bind(i, j, k)
              C[i0, i1, i2] = (2 * (1 + A[i1, i0, i2]))
            }
          }
        }
      }
    }
  }
```

可以看到，原先有两个 ijk 的循环， ComputeInline 之后，合并成了一个循环，并且消除了中间变量 B[i1, i0, i2]。

ComputeInline 的顶层实现为。

```C++
void StaticShapeGroupScheduler::DoComputeInline() {
  VLOG(5) << "[Start DoComputeInline] func body: "
          << ir_sch_->GetModule().GetExprs().front();

  std::unordered_set<std::string> no_inline_output_names = OutputTensorNames();
  auto_schedule::AutoInline inliner(target_, no_inline_output_names);

  auto InlineFunc = [&](ir::ScheduleBlockNode* node) {
    if (IsProhibitScheduleExternCallBlock(node->Block())) {
      return;
    }
    VLOG(6) << "try ComputeInline on: " << node->id()
            << ", before ComputeInline, func body: "
            << ir_sch_->GetModule().GetExprs().front();
    ir::Expr schedule_block = node->Block();
    inliner.Apply(ir_sch_, schedule_block);
    VLOG(6) << "try ComputeInline on: " << node->id()
            << ", after ComputeInline, func body: "
            << ir_sch_->GetModule().GetExprs().front();
  };

  schedule_block_graph_->DFSTopoWalk(InlineFunc);
  schedule_block_graph_->Update(*ir_sch_);
  VLOG(5) << "[After DoComputeInline] func body: "
          << ir_sch_->GetModule().GetExprs().front();
}
```

主要步骤
- 获取不能内联化的 tensor names
- 生成 AutoInline
- schedule_block_graph_ 以拓扑排序使得各个节点调用 InlineFunc
- 更新 ir_sch_

其中核心为 InlineFunc，对符合条件的 node->Block() 应用 inliner.Apply(ir_sch_, schedule_block)。AutoInline::Apply 内部除去一些验证相关的操作，
就是调用 ir_schedule->ComputeInline(block_expr)，内联至 Consumer （即消除第一个 ijk 循环，并将其合并到第二个 ijk 循环）。

ComputeInline(const Expr& schedule_block) 有两种实现方式，动态实现与静态实现，我们主要关注静态实现。

```C++
void StScheduleImpl::ComputeInline(const Expr& schedule_block) {
  CHECK(schedule_block.As<ir::ScheduleBlockRealize>());
  Expr root = this->GetRootBlock(schedule_block);
  Expr store = CheckComputeInlineValidationAndGetStore(schedule_block, root);
  ComputeInliner inliner(store.As<ir::Store>()->tensor.as_tensor_ref(), store);
  CHECK(inliner.BodyPatternAllowInline());
  // Create a plan that removes the block to be inlined
  LeafBlockRemovalPlan remove_plan(
      schedule_block, &inliner.src_stmt, &inliner.tgt_stmt);
  remove_plan(&root);
  inliner(&root);
  return;
}
```

主要步骤
- 获取 root
- 获取 store
- 去除即将内联化的 block
- 计算内联化

TODO：root 的作用及实现过程

TODO：store 的作用及实现过程

我们接下来重点分析 LeafBlockRemovalPlan 与 ComputeInliner，探究一下实现细节。

## 3. LeafBlockRemovalPlan

TODO

## 4. ComputeInliner

由上文 inliner(&root) 及相关源码可知， ComputeInliner 的父类重载了 () 运算符。

```C++
void BaseInliner::operator()(Expr* expr) {
  IRMutator::Visit(&tgt_stmt, &tgt_stmt);
  IRMutator::Visit(expr, expr);
}
```

TODO：为什么 Visit 要传两个一样的参数?

TODO: IRMutator::Visit(&tgt_stmt, &tgt_stmt), IRMutator::Visit(expr, expr) 分别都有什么作用？

重载函数调用了 visit 函数。而 ComputeInliner 又重写了 Visit(const ir::Load* expr, Expr* op)。

```C++
void ComputeInliner::Visit(const ir::Load* expr, Expr* op) {
  if ((expr->tensor).as_tensor_ref()->name == inlined_tensor_->name) {
    *op = ReplaceInlinedTensor(op);
    return;
  }
  IRMutator::Visit(expr, op);
}
```

由 [背景知识](#1-背景知识) 可知，父类 Visitor 以 dfs 的顺序遍历。子类调用父类的 visit 函数，也会以 dfs 的顺序遍历。
故而，上面代码的逻辑即为：在未遍历到要做内联化的节点时（节点的类型不是 ir::Load* 或者 (expr->tensor).as_tensor_ref()->name ！= inlined_tensor_->name），以 dfs 的顺序遍历；
遍历到要做内联化的节点后，调用 ReplaceInlinedTensor 函数，并直接返回。

```C++
//! Replace the 'Load' node on the tensor to 'Load' node of its producers.
Expr ComputeInliner::ReplaceInlinedTensor(Expr* load) {
  CHECK(load->As<ir::Load>());
  SetIndexSubstitution(load->As<ir::Load>()->indices);
  Expr value_copy = ir::ir_utils::IRCopy(inlined_store_.As<Store>()->value);
  ReplaceExpr(&value_copy, idx_sub_var_, idx_sub_expr_);
  return value_copy;
}
```

主要步骤
- 设置 idx_sub_var_， idx_sub_expr_（SetIndexSubstitution）
- 复制一份 Expr
- 替换

```C++
void BaseInliner::SetIndexSubstitution(const std::vector<Expr>& indices) {
  CHECK_EQ(indices.size(), idx_vars_.size());
  int n = idx_vars_.size();
  idx_sub_var_.reserve(n);
  idx_sub_expr_.reserve(n);
  for (int i = 0; i < n; ++i) {
    idx_sub_var_.push_back(idx_vars_[i]);
    idx_sub_expr_.push_back(indices[i]);
  }
}
```

需要注意的是，idx_vars_ 有注释：The indices used for indexing the buffer to be inlined 。可见 idx_sub_var_ 对应要被内联消除的索引信息，
idx_sub_expr_ 对应 consumer 的索引信息。

```C++
Expr IRCopy(Expr x, bool copy_buffer_node) {
  IRCopyVisitor visitor(copy_buffer_node);
  auto copied = visitor.Visit(&x);
  return copied;
}
```

调用 IRCopyVisitor::Visit 函数。IRCopyVisitor 重写了很多入参类型不同的 Visit 函数。作用均为复制该节点的数据并返回。

但是，这里调用的是哪一个呢？

从上文的 void ComputeInliner::Visit(const ir::Load* expr, Expr* op) 可知， Visit 的返回值 copied 经过层层传递，
最终赋值给了 *op ，而 op 与 expr 类型一致，为 `ir::Load*` 。所以这里推断调用的是：

```C++
  Expr Visit(const Load* op) override {
    auto tensor = Visit(&op->tensor);
    std::vector<Expr> indices;
    for (auto& idx : op->indices) {
      indices.push_back(Visit(&idx));
    }
    return Load::Make(tensor, indices);
  }
```

通过 Visit 遍历子节点，获取复制的 tensor 与 indices，封装成 Load 并返回。再看 tensor 是如何复制的。

```C++
  Expr Visit(const _Tensor_* op) override {
    if (tensor_map.count(op->name)) {
      return tensor_map[op->name];
    }

    auto shape = Visit(op->shape);
    auto domain = Visit(op->domain);
    auto buffer_expr = Expr(op->buffer);
    // TODO(Superjomn) copy the operation.
    auto operaion = op->operation;
    auto name = op->name;
    auto tensor = make_shared<_Tensor_>();

    // tensor->buffer = op->buffer;
    if (buffer_expr.defined()) {
      if (copy_buffer_node) {
        auto buffer = Visit(&buffer_expr);
        tensor->buffer = buffer.as_buffer_ref();
      } else {
        tensor->buffer = op->buffer;
      }
    }
    tensor->domain = domain;
    tensor->shape = shape;
    tensor->reduce_axis = op->reduce_axis;
    tensor->operation = operaion;
    tensor->name = name;
    tensor->set_type(op->type());
    tensor->axis_ = op->axis_;

    tensor_map[tensor->name] = tensor;

    return tensor;
  }
```

如果 tensor_map 中缓存有 op->name 对应的tensor，则直接返回。否则通过 Visit 遍历子节点组装返回。

我们再回过头来，看一下 ReplaceExpr 是如何实现的。

```C++
void ReplaceExpr(Expr* source,
                 const std::vector<Var>& replaced,
                 const std::vector<Expr>& candidates) {
  CHECK_EQ(replaced.size(), candidates.size())
      << "In ReplaceExpr, the size of Vars to be replaced must be equal to the "
         "size of candidate Exprs! Please check.";
  if (replaced.empty()) return;
  std::map<Var, Expr, CompVar> replacing_map;
  for (int i = 0; i < replaced.size(); ++i) {
    // If the Var to be replaced is equal to the candidate, we skip it.
    if (candidates[i].is_var() && candidates[i].as_var_ref() == replaced[i])
      continue;
    replacing_map[replaced[i]] = candidates[i];
  }
  MappingVarToExprMutator mapper(replacing_map);
  mapper(source);
  return;
}
```

构建 replacing_map ，建立 replaced 与 candidates的一一映射关系。然后通过 MappingVarToExprMutator 调用 Visit 函数。

```C++
  void Visit(const ir::_Var_* expr, Expr* op) override {
    if (replacing_map_.count(op->as_var_ref())) {
      *op = replacing_map_.at(op->as_var_ref());
    }
  }
```

替换 value_copy 子节点的索引信息。
