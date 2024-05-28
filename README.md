# GrabCut
my implementation of grabcut

除了border matting都完成了，后续可能会更新这部分。

个人博客的论文精读和实现指南：[https://kegalas.top/p/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB%E4%B8%8E%E5%A4%8D%E7%8E%B0/](https://kegalas.top/p/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB%E4%B8%8E%E5%A4%8D%E7%8E%B0/)

原论文：[https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/](https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/)

requirements:
- opencv-2.4.13.6
- cmake-3.27.6
- g++-13.1.0
- maxflow-3.01

其中maxflow在代码中已经有了，原始文件可以在[https://vision.cs.uwaterloo.ca/code/](https://vision.cs.uwaterloo.ca/code/)找到
