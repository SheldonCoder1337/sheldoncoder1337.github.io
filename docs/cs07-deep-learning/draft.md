# How I got into deep learning

I ran an education company, [Dataquest](https://www.dataquest.io/), for 8 years. Last year, I got the itch to start building again. Deep learning was always interesting to me, but I knew very little about it. I set out to fix that problem.

Since then, I’ve trained dozens of [models](https://huggingface.co/vikp) (several state of the art for open source), built [2 libraries](https://github.com/VikParuchuri/) that have 5k+ Github stars, and recently accepted an offer from answer.ai, a research lab started by Jeremy Howard.

I say this to establish the very rough outline of my learning journey. In this post, I’m going to cover more detail about how I learned deep learning. Hopefully it helps on your journey.

## My background

I didn’t learn this stuff in school. I majored in American History for undergrad, and [failed quite a few classes](https://www.vikas.sh/post/i-barely-graduated-college).

I did machine learning and Python work in 2012, but convinced myself that deep learning was too complicated for me. One reason for this was because I learned by doing Kaggle competitions. Kaggle competitions are amazing for learning quickly, but can leave you with gaps in the fundamentals - like how algorithms work mathematically.

When deep learning started to become popular, it was very math-heavy, and I felt like I’d never be able to understand it. Of course, this was false, as I proved to myself 10 years later, but the angle at which you approach something makes all the difference. I approached deep learning top-down the first time, by gluing models together without understanding how they worked. I eventually hit a wall, and couldn’t get past it.

## Useful skills

When I studied deep learning last year, I already had useful skills. The first was strong Python programming ability. Despite efforts to the contrary, Python is still the universal language of AI. If you want to get into AI, start by getting really good at programming.

No matter what era of AI I’ve been in, data cleaning has been >70% of my work. It’s possible if you’re doing pure research or working on toy problems you can avoid working with data, but otherwise data skills are essential.

There’s a slightly more nebulous skill I’ll call pragmatism. Deep learning has a lot of rabbit holes - ranging from “what’s the perfect base model?”, to “what if I get rid of the sigmoid here?” Some of these rabbit holes are useful, but most of them will eat a lot of time. Being able to recognize when to go deep, and when to just do the fast/easy solution is important.

## Book learning

This time, I decided to learn bottom-up, fundamentals first. I read [The Deep Learning Book](https://www.deeplearningbook.org/). It’s a few years old, but still a fantastic resource. Read it slowly. A lot of the terminology and math will be unfamiliar - look them up. You may need to sketch some things out or code them to get them - give yourself the space to do that. If the math is unfamiliar, a good complementary resource is [Math for Machine Learning](extension://bfdogplmndidlpjfhoijckpakkdjkkil/pdf/viewer.html?file=https%3A%2F%2Fmml-book.github.io%2Fbook%2Fmml-book.pdf). Although I haven’t taken them, [fast.ai](https://www.fast.ai/) and the [Karpathy](https://www.youtube.com/@AndrejKarpathy) videos are high quality.

Even though architectures like CNN or RNN might seem out of date in a world that is moving towards transformers for everything, CNNs are still widely used, and everything old is new again with RNNs.

When you’re done with the first 2 parts of the book (you can skip part 3), you should be at a point where you can code up any of the main neural networks architectures in plain numpy (forward and backward passes).

One thing that will really help you get to that point is teaching the skills while you learn them. I started putting together a course, [Zero to GPT](https://github.com/VikParuchuri/zero_to_gpt), as I read the deep learning book. Teaching is the ultimate way to solidify concepts in your head, and I found myself falling into a nice cycle of learning, looking up/sketching what I didn’t understand, then teaching it.

## Papers

The book will take you up to 2015-era deep learning. After reading the book, I read some of the foundational deep learning papers from the 2015-2022 era and implemented them in PyTorch. You can use Google Colab for free/cheap GPUs, and Weights and Biases to track your training runs.

A noncomprehensive list is:

- RNN attention
- Transformers
- Switch transformer
- LoRA
- Vision Transformer
- AdamW
- GPT-2

After this, you should be able to understand most conversations people have about deep learning model architectures.

## Fine-tuning and Discord

The easiest entrypoint for training models these days is finetuning a base model. [Huggingface transformers]() is great for finetuning because it implements a lot of models already, and uses PyTorch.

There are Discord communities, like Nous Research and EleutherAI where people discuss the latest models and papers. I’d recommend joining them, seeing what’s state of the art at the moment, and trying some finetuning.

The easiest way to finetune is to pick a small model (7B or fewer params), and try finetuning with LoRA. You can use Google Colab, or something like Lambda Labs if you need more VRAM or multiple GPUs.

I wanted to train models to code better, so I put together datasets and finetuned a few different base models on data from StackOverflow and other places. It really helped me understand the linkage between model architecture, data, compute, and output. However, finetuning is a very crowded space, and it’s hard to make an impact when the state of the art changes every day.

## Problem Discovery

As I was working on finetuning, I realized that some of the highest quality data was in textbook form, and locked away in pdfs. One way I tried to solve this was to generate [synthetic data].

Another way was to extract the data from pdfs and turn it into good training data (markdown). There was an approach called [nougat] that worked well in many cases, but was slow and expensive to run. I decided to see if I could build something better by leveraging the data already in the pdf (avoiding OCR), and only using models when needed. I chained together several different models, along with heuristics in between. This approach, [marker](https://github.com/datalab-to/marker), is 10x faster than nougat, works with any language, and is usually more accurate.

Working on marker led me to want to solve several more problems, and I’ve also trained an equation to [LaTeX model (texify)](https://github.com/VikParuchuri/texify), a [text detection model](), an [OCR model (surya)](https://github.com/datalab-to/surya) that’s competitive with Google Cloud, and a layout model.

For all of these models, I took existing architectures, changed the layers, loss, and other elements, then generated/found the right datasets. For example, for the OCR model, I started with the Donut architecture, added GQA, an MoE layer, UTF-16 decoding (1-2 tokens for any character), and changed some of the model shapes.

Since OCR models are typically small (less than 300M params), I was able to train all of these models on 4x A6000s. I probably could have gotten away with 2x A6000s if I was a bit more efficient.

Hopefully this illustrates 3 things for you:

- Understanding the fundamentals is important to training good models
- Finding interesting problems to solve is the best way to make an impact with what you build
- You don’t need a lot of GPUs

There are many niches in AI where you can make a big impact, even as a relative outsider.

## Open source

As you may have noticed, I open source all of my AI projects. The data stack is a very underinvested area of AI relative to impact. I feel strongly that the more widely you can distribute high quality training data, the lower the risk of 1-2 organizations having a monopoly on good models.

Open source also has a side effect of being a good way to get exposure. Which leads me to the last part of my story.

## Getting a research job

I was thinking about building a business around my open source tools. Working somewhere wasn’t on my radar at all. But when Jeremy reached out about answer.ai, I felt like it was an opportunity I had to take. The chance to work with talented people, make a positive impact, and learn a lot is hard to pass up.

My open source work directly led to the job opportunity, both in the obvious way (it gave me exposure), and in a subtler way (it significantly improved my skills). Hopefully, you’ll open source something as you learn, too.

## Next steps

I suspect my work at answer.ai will look very similar to my open source work. I’ll keep training models, improving the data stack, and releasing publicly.

If you’re trying to break into deep learning, I hope this post was useful. If not, I hope it was somewhat entertaining (you made it to the end, so it probably was, right?).

As for me, I’m going back to training some models (watching 2 of them converge right now).

## About Data

总共有5个模型

1. 文本布局检测 layout
2. OCR错误检测
3. 表格识别 table_recognition
4. 文本检测 text_detection
5. 文本识别 text_recognition

### OCR模型数据集（基于Donut架构改进）

作者在OCR模型中引入GQA（门控注意力）、MoE（专家混合层）、UTF-16解码等创新，需支持多语言、复杂排版的数据。

公开数据集

1. SynthText：合成自然场景文本图像（85万张），含单词/字符级边界框标注，支持多字体、扭曲文本和复杂背景生成4。
2. ICDAR系列：如ICDAR2015，包含自然场景文本检测与识别数据，适用于端到端OCR训练8。

自定义生成方法

- 字体渲染+背景合成：
  - 从语料库随机抽取文本（如5–10个字符），结合多种字体渲染；
  - 对背景图聚类分析，选择与背景对比度最高的文字颜色（如从500种颜色中筛选对比度最高的200种）9；
  - 添加透视变换、高斯模糊等增强，模拟真实场景退化9。
- 工具链支持：
  - 使用SynthText生成代码批量合成文本图像4；
  - 中文场景可参考生成3755个汉字的印刷体数据集（支持笔画粘连、旋转等12种增强）5。

!!! note "可能使用的数据集："
        所用架构：基于 Donut（OCR-free Transformer 架构），加入了 GQA、MoE、UTF-16 解码等模块。
        可能使用的数据集：
        - CORD-v2（用于训练 Donut 原始模型）
        - Post-OCR Correction Dataset（PleIAs, 2023）：包含 Chronicling America 的报纸图像和对应的 OCR 错误/修正文本，适合训练 OCR 纠错模型
        - Noisy OCR Dataset（Kaggle 等来源）：用于训练去噪或提升 OCR 鲁棒性
        如何生成：
        - 使用合成数据工具（如 SynthText、TrOCR 的 SynthDoG）生成带文本的图像；
        - 或使用真实扫描文档+OCR引擎（Tesseract、EasyOCR）生成伪标签，再人工清洗。

### 文本检测模型数据集

需高精度文本框标注，支持复杂布局（如倾斜文本、密集排列）。

1. 公开数据集
   - ICDAR2015：提供自然场景文本的定位标注8；
   - MSRA-TD500：多语言、多方向文本检测基准。
2. 自定义生成与标注
   - 半自动标注工具：
     - 使用PPOCRLabel标注工具，手动框选文本区域生成标签文件（支持PaddleOCR格式）8；
   - 合成数据混合：
     - 将SynthText合成数据与真实数据混合训练，提升模型泛化性4。

!!! note "可能使用的数据集："
        - COCO-Text：包含自然场景图像中的文本框标注；
        - ICDAR 系列（ICDAR 2013/2015/2019）；
        - SynthText（合成文本检测数据）；
        - WildReceipt：收据场景文本检测数据

### 布局模型数据集

作者需生成文档/图像元素的边界框与语义标签（如标题、图表位置）。

1. 公开数据集
   - LayoutSAM：复旦与字节推出的270万图像-文本对数据集，含1070万个实体标注（颜色、形状、纹理等属性）3；
   - PubLayNet：科研文献的布局标注数据集。

2. 自定义生成方法
   - 布局合成技术：
     - CreatiLayout：基于文本描述或草图生成布局（如海报、家具摆放），通过LayoutDesigner模块优化元素位置3；
     - Layout-Generator工具：生成热源组件布局-温度场数据集，支持矩形组件离散/连续采样（如Gibbs采样）7。

!!! note "可能使用的数据集："
        - PubLayNet：来自PubMed论文的布局标注（标题、段落、图表等）；
        - DocBank：文档图像+布局框标注；
        - Fintabnet（IBM）：用于表格识别，包含行列结构；
        - 自建数据集：使用像 Marker 这样的工具生成带布局标签的PDF图像，或从 arXiv 论文中提取页面并人工标注阅读顺序。

``` mermaid
graph LR
A[输入草图/文本描述] --> B(CreatiLayout生成布局)
B --> C[输出元素坐标+语义标签]
C --> D[存储为HDF5或PNG掩码]
```

### 公式

需数学公式图像与LaTeX序列的配对数据。

1. 公开数据集
   - Im2Latex-100K：10万公式图像-LaTeX对照数据集；
   - MathPix：API可生成公式截图与LaTeX标注。
2. 自定义生成方法
   - 合成流程：
      1. 从LaTeX公式库随机采样序列；
      2. 渲染为PNG图像（添加字体变形、噪声、分辨率变化）；
      3. 生成对抗样本：旋转公式、叠加文档背景纹理59。
   - 工具推荐：
     - Python库PIL渲染公式 + augmentor添加畸变。

!!! note "可能使用的数据集："
        - im2latex-100k：包含约100k张公式图像与对应LaTeX代码；
        - arxiv-Latex：从 arXiv 论文中提取的公式图像与LaTeX；
        - 自建数据集：通过 Mathpix、LaTeX-OCR 等工具辅助标注，或从公开论文中爬取并清洗。

五、作者技术适配与数据集选择建议

作者对Donut架构的改进（如MoE、UTF-16解码）要求数据集满足：

- 多语言支持：生成时需涵盖UTF-16字符集（如中日韩文字），可用5方法扩展字符集；
- 高复杂度：通过合成数据模拟模糊、倾斜文本，匹配MoE层对困难样本的处理需求9；
- 布局多样性：使用CreatiLayout生成多模态布局数据，适配GQA的跨模态注意力机制3。

``` python
# OCR数据生成示例（结合背景与字体增强）
from font_renderer import render_text
from augmentor import add_noise, perspective_transform

text = sample_corpus()  # 从语料库采样
image = render_text(text, font="random", color="high_contrast")  
image = add_noise(image, type="gaussian")  
image = perspective_transform(image, max_angle=15)  
save_data(image, label=text) 
```

作者的数据集策略分为三类：

1. 公开数据集：ICDAR（文本检测）、LayoutSAM（布局）、SynthText（OCR合）；
2. 半自动标注：PPOCRLabel（真实数据标注）；
3. 全合成生成：字体渲染+背景融合（OCR）、LaTeX渲染（公式）、CreatiLayout（布局）。

其创新点（如MoE、UTF-16）依赖合成数据的可控性与多样性，开源工具SynthText、CreatiLayout和自定义渲染脚本是核心解决方案。