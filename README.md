# Paddle-Code-Camp

## 相关链接

百度飞桨护航计划
- https://github.com/PaddlePaddle/Paddle/issues/61006

项目要求
- [CINN 静态 shape 下鲁棒性和性能优化](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/%E3%80%90Hackathon%206th%E3%80%91%E9%A3%9E%E6%A1%A8%E6%8A%A4%E8%88%AA%E8%AE%A1%E5%88%92%E9%9B%86%E8%AE%AD%E8%90%A5%E9%A1%B9%E7%9B%AE%E5%90%88%E9%9B%86.md#%E9%A1%B9%E7%9B%AE%E5%8D%81%E5%85%ABcinn-%E9%9D%99%E6%80%81-shape-%E4%B8%8B%E9%B2%81%E6%A3%92%E6%80%A7%E5%92%8C%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96)

## 背景

编译器一直是深度学习框架的深水区，同时也是厂商竞争的必争之地。主流的深度学习框架都会利用编译技术直接将多个Op自动生成Kernel，可以节省大量的GPU GlobalMemory的读写时间。Paddle拥有自己的CINN编译器，这个专项是针对CINN的静态Shape部分进行性能优化和架构重新调整，目的是提高CINN的鲁棒性和Kernel生成性能。

这个任务主要是针对CINN中不太高效或者是不太规范的部分，进行代码重构和优化，包含下面2个方面：
1. 鲁棒性，我们需要确保我们的修复和策略是在正确的道路上，而不是case by case的修复。这个部分我们通过构建了完善的子图验证集合来保证。
2. 性能，我们的其他部分的优化会针对性能，因此掌握cuda的编程，并看到最终的Kernel并找到最终的Kernel的问题是很重要的能力。当然作为实习生，可能只需要关注如何将一个优化实现即可。如果可以自主找到待优化点并取得性能收益，那就更好了。

## 项目规划和MileStone

MileStone1：熟悉CINN前端后端流程（CodeGen之前），并具备独立修复CINN中小bug的能力。
成果交付：
1. 产出CINN的流程图，并给导师进行技术分享。
2. 独立修复CINN的框架 bug 5个。**后续会按照出错的子图，进行分配。**
3. 熟悉基本的cuda编程，具备看懂cuda代码的能力。文档可以见：https://github.com/PaddleJitLab/CUDATutorial 

MileStone2：参与GroupSchedule重构的开发，规范CINN行为，增加鲁棒性。
要求：并能独立解决分配的任务。
成果交付：【待设定】

MileStone3：参与性能优化，能够将优化方案落实到CINN的代码中。
成果交付：【待设定】

年前任务
MileStone1
