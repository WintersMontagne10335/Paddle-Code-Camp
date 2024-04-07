# 一、 ComputeInline

- [一、ComputeInline](#一-ComputeInline)
  - [1. 背景知识](#1-背景知识)
  - [2. 主要流程](#2-主要流程)
  - [3. LeafBlockRemovalPlan](#3-LeafBlockRemovalPlan)
  - [4. ComputeInliner](#4-ComputeInliner)

## 1. 背景知识

### Ⅰ. 访问者（Visitor） 模式
访问者模式是一种行为型设计模式，它将算法与其所作用的对象分离开来，使得能够在不改变对象结构的前提下，对对象中的元素进行新的操作。
该模式的核心思想是，定义一个访问者对象，并将其传递给需要被访问的对象，在对象接受访问者的访问时，会调用访问者对象中的方法，在该方法中实现对象对于访问者的响应操作。

以下图为例。

![image](https://github.com/WintersMontagne10335/Paddle-Code-Camp/assets/118546135/83ade3bc-1e76-4a0d-a63c-504d00f7990a)

首先，对于上面每个地方，每个人都可能有不同的访问行为，有游客到济公殿会看看墙上济公的故事，有的游客可能就对济公殿很熟悉，
进去拜一下就出来了，有的到了断桥残雪感慨一句全是人头，也有人到了断桥残雪，就会会回味一下张祜的诗句。

其次，对于上面每个地方，对于不同的游客来说，不确定，所以，不管是谁，给他准备个入口进来总不会有问题。

结合上面分析，我们可以初步分析访问者模式是怎么一回事儿：在每个游客自己内部，把对每个景点想操作的行为定义出来，然后在每个景点，准备个接口，把游客迎进来，
最后呢，把这个游客迎进来之后呢，执行这个游客里面定义的，关于自己的行为。

实现方式上，一般为，基类(BaseVisitor)含有所有节点（景点）的 visit 方法，这个 visit 方法什么都不做，只是按 dfs 的顺序调用其它节点（景点）的 visit 方法。
子类 visitor 继承基类，重写某些节点的 visit 方法（比如张三是子类 visitor，他到了断桥残雪，就会会回味一下张祜的诗句）。在子类 visit 方法的结尾，
如果想继续以 dfs 的顺序遍历，就应该调用基类的 visit 方法，再 return；如果想直接返回，直接 return 即可。

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

我们接下来重点分析 LeafBlockRemovalPlan 与 ComputeInliner，探究一下实现细节。

## 3. LeafBlockRemovalPlan

TODO

## 4. ComputeInliner

TODO

