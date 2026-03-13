# Markdown 基本语法

本演示手册依照 [Material for mkdocs](https://squidfunk.github.io/mkdocs-material/reference/) 的官方文档制作，目的是记录未来可能会用到的 Markdown 格式，本文中的许多 Markdown 格式均不具有普适性，仅可在 pymdownx 扩展支持下的 Material for mkdocs 中生效。以下是基于官方文档所制作的中文版参考文档，你可以参考这些文档来书写自己的 .md 文档，进而构建到网站上。由于种种原因，内容可能不尽完全。

## Admonitions 警示块

---

警示框，又称提示框，是一种非常适合在不中断文档流的情况下包含附加内容的工具。Material for MkDocs 提供了多种类型的警示框，并允许在其中嵌套任意内容。

### Configuration

```yaml title="mkdocs.yml"
markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
```

### Usage

```text
!!! note
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.
```

!!! note
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.

```text
自定义标题
!!! note "Phasellus posuere in sem ut cursus"
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 

嵌套
!!! note "Outer Note"
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
    !!! note "Inner Note"
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. 

无标题
!!! note ""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. 

折叠
??? note
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.

折叠且初始化展开
???+ note
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.

内联块
!!! info inline end "Lorem ipsum" 或者 info inline

    Lorem ipsum dolor sit amet, consectetur
    adipiscing elit. 
```

note, abstract, info, tip, success, question, warning, failure, danger, bug, example, quote

!!! info inline end "Lorem ipsum"

    Lorem ipsum dolor sit amet, consectetur
    adipiscing elit. 

??? example

    Lorem ipsum dolor sit amet, consectetur adipiscing elit.

## Annotations 注解

---

Material for MkDocs 的旗舰功能之一是能够注入注释——几乎可以在文档中的任何位置添加的小标记，并在单击或键盘焦点时扩展包含任意 Markdown 的工具提示。

### Configuration

```yaml title="mkdocs.yml"
theme:
  icon:
    annotation: material/arrow-right-circle

markdown_extensions:
  - attr_list
  - md_in_html
  - pymdownx.superfences
```

### Usage

```text title="markdown.md"
!!! note annotate "Phasellus posuere in sem ut cursus (1)"
    Lorem ipsum dolor sit amet, (2) consectetur adipiscing elit.
    { .annotate }

1. :man_raising_hand: I'm an annotation! (1)
    { .annotate }
    1. :woman_raising_hand: I'm an annotation as well!

2. :man_raising_hand: I'm an annotation!
```

!!! note annotate "Phasellus posuere in sem ut cursus (1)"
    Lorem ipsum dolor sit amet, (2) consectetur adipiscing elit.
    { .annotate }

1. :man_raising_hand: I'm an annotation! (1)
    { .annotate }
    1. :woman_raising_hand: I'm an annotation as well!

2. :man_raising_hand: I'm an annotation!

## Buttons 按钮

---

### Configuration

```yaml title="mkdocs.yml"
markdown_extensions:
  - attr_list
```

### Usage

```text title="markdown.md"
[Send :fontawesome-solid-paper-plane:](#){ .md-button }
```

[Send :fontawesome-solid-paper-plane:](#){ .md-button }

## Code blocks 代码块

---

### Configuration

```yaml title="mkdocs.yml"

theme:
  features:
    - content.code.copy
    - content.code.select
    - content.code.annotate

markdown_extensions:
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences

extra:
  annotate:
    json: [.s2]
```

### Usage

```cpp title="代码块"
#include <iostream>
int main()
{
    std::cout<<"Hello world!";
    return 0; // (1)
}   // (2)!
```

1. 注解必须放在注释语句内，而插入代码中的注解是 Insiders 特有的功能。
2. 如果你想显示这个注释符号(.cpp中是`//`)，那么删去序号后面的`!`即可。

``` py title="行编号" linenums="1"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```

=== "highlight 2 3"

    ``` py title="行高亮" hl_lines="2 3"
    def bubble_sort(items):
        for i in range(len(items)):
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
    ```

=== "highlight 3-5"

    ``` py title="行高亮" hl_lines="3-5"
    def bubble_sort(items):
        for i in range(len(items)):
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
    ```
    
## Content tabs 选项卡

---

有时，将不同的内容分组到不同的选项卡下是很有用的，例如描述如何在不同语言下编写功能相同的代码时。Material for MkDocs 提供了美观且实用的选项卡功能，可以将代码块和其他的内容分组显示。

### Configuration

```yaml title="mkdocs.yml"
theme:
  features:
    - content.tabs.link # 启用后，当用户单击选项卡时，整个文档站点中的所有内容选项卡都将链接并切换到相同的标签

markdown_extensions:
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true # 此配置启用内容选项卡，并允许在内容选项卡中嵌套任意内容，包括代码块和...更多内容选项卡！
```

### Usage

```text title="Code blocks"
=== "C"

    ``` c
    #include <stdio.h>

    int main(void) {
      printf("Hello world!\n");
      return 0;
    }
    ```

=== "python"

    ``` python
    print("Hello world!\n")
    ```
```

=== "C"

    ``` c
    #include <stdio.h>

    int main(void) {
      printf("Hello world!\n");
      return 0;
    }
    ```

=== "python"

    ``` python
    print("Hello world!\n")
    ```

```text title="Content blocks"
=== "Unordered list"

    * Sed sagittis eleifend rutrum
    * Donec vitae suscipit est
    * Nulla tempor lobortis orci

=== "Ordered list"

    1. Sed sagittis eleifend rutrum
    2. Donec vitae suscipit est
    3. Nulla tempor lobortis orci
```

=== "Unordered list"

    * Sed sagittis eleifend rutrum
    * Donec vitae suscipit est
    * Nulla tempor lobortis orci

=== "Ordered list"

    1. Sed sagittis eleifend rutrum
    2. Donec vitae suscipit est
    3. Nulla tempor lobortis orci

```markdown title="内容嵌套"
!!! example

    === "Unordered List"

        ``` markdown
        * Sed sagittis eleifend rutrum
        * Donec vitae suscipit est
        * Nulla tempor lobortis orci
        ```

    === "Ordered List"

        ``` markdown
        1. Sed sagittis eleifend rutrum
        2. Donec vitae suscipit est
        3. Nulla tempor lobortis orci
        ```
```

!!! example

    === "Unordered List"

        ``` markdown
        * Sed sagittis eleifend rutrum
        * Donec vitae suscipit est
        * Nulla tempor lobortis orci
        ```

    === "Ordered List"

        ``` markdown
        1. Sed sagittis eleifend rutrum
        2. Donec vitae suscipit est
        3. Nulla tempor lobortis orci
        ```

## Data Tables 数据表

---

MkDocs 的 Material 定义了数据表的默认样式——这是在项目文档中呈现表格数据的绝佳方式。此外，可以使用第三方库和一些额外的 JavaScript 来实现可排序表格等自定义。

### Configuration

```yaml title="mkdocs.yml"
markdown_extensions:
  - tables
```

### Usage

```text title="数据表"
| Method      | Description                          |
| ----------- | ------------------------------------ |
| :---------- | :----------------------------------- | 列左对齐
| :---------: | :----------------------------------: | 列中对齐
| ----------: | -----------------------------------: | 列右对齐
| `GET`       | :material-check:     Fetch resource  |
| `PUT`       | :material-check-all: Update resource |
| `DELETE`    | :material-close:     Delete resource |
```

| Method      | Description                          |
| ----------- | ------------------------------------ |
| `GET`       | :material-check:     Fetch resource  |
| `PUT`       | :material-check-all: Update resource |
| `DELETE`    | :material-close:     Delete resource |

## Diagrams 图流

---

图表有助于传达不同技术组件之间的复杂关系和互连，是项目文档的重要补充。Material for MkDocs 与 Mermaid.js 集成，这是一种非常流行且灵活的图表绘制解决方案。

### Configuration

```yaml title="mkdocs.yml"
markdown_extensions:
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
```

### Usage

``` mermaid title="流程图"
graph LR
  A[Start] --> B{Error?};
  B -->|Yes| C[Hmm...];
  C --> D[Debug];
  D --> B;
  B ---->|No| E[Yay!];
```

``` mermaid title="序列图"
sequenceDiagram
  autonumber
  Alice->>John: Hello John, how are you?
  loop Healthcheck
      John->>John: Fight against hypochondria
  end
  Note right of John: Rational thoughts!
  John-->>Alice: Great!
  John->>Bob: How about you?
  Bob-->>John: Jolly good!
```

``` mermaid title="状态图"
stateDiagram-v2
  state fork_state <<fork>>
    [*] --> fork_state
    fork_state --> State2
    fork_state --> State3

    state join_state <<join>>
    State2 --> join_state
    State3 --> join_state
    join_state --> State4
    State4 --> [*]
```

``` mermaid title="类图"
classDiagram
  Person <|-- Student
  Person <|-- Professor
  Person : +String name
  Person : +String phoneNumber
  Person : +String emailAddress
  Person: +purchaseParkingPass()
  Address "1" <-- "0..1" Person:lives at
  class Student{
    +int studentNumber
    +int averageMark
    +isEligibleToEnrol()
    +getSeminarsTaken()
  }
  class Professor{
    +int salary
  }
  class Address{
    +String street
    +String city
    +String state
    +int postalCode
    +String country
    -validate()
    +outputAsLabel()  
  }
```

``` mermaid title="实体关系图"
erDiagram
  CUSTOMER ||--o{ ORDER : places
  ORDER ||--|{ LINE-ITEM : contains
  LINE-ITEM {
    string name
    int pricePerUnit
  }
  CUSTOMER }|..|{ DELIVERY-ADDRESS : uses
```

除了上面列出的图表类型外，Mermaid.js 还支持饼图 、 甘特图 、 用户旅程 、git 图和需求图 ，所有这些都不受 MkDocs 材料的官方支持。这些图表应该仍然像 Mermaid.js 宣传的那样工作，但我们认为它们不是一个好的选择，主要是因为它们在移动设备上效果不佳。

## Footnotes 脚注

---

脚注是向特定单词、短语或句子添加补充或附加信息的好方法，而不会中断文档的流程。MkDocs 的 Material 提供了定义、引用和渲染脚注的能力。

### Configuration

```yaml title="mkdocs.yml"
theme:
  features:
    - content.footnote.tooltips

markdown_extensions:
  - footnotes
```

### Usage

```text title="Footnotes demo"
Lorem ipsum[^1] dolor sit amet, consectetur adipiscing elit.[^2]

[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
[^2]:
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.
```

Lorem ipsum[^1] dolor sit amet, consectetur adipiscing elit.[^2]

[^1]: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
[^2]:
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod
    nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor
    massa, nec semper lorem quam in massa.

## Formatting 文本元素格式化

---

MkDocs-material 支持多个 HTML 元素，这些元素可用于突出显示文档的各个部分或应用特定格式。此外，还支持 Critic Markup，从而增加了显示文档建议更改的功能。

### Configuration

```yaml title="mkdocs.yml"
markdown_extensions:
  - pymdownx.critic
  - pymdownx.caret
  - pymdownx.keys
  - pymdownx.mark
  - pymdownx.tilde
```

### Usage

```text title="Formatting Demo"
高亮显示更改
Text can be {--deleted--} and replacement text {++added++}. This can also be
combined into {~~one~>a single~~} operation. {==Highlighting==} is also
possible {>>and comments can be added inline<<}.

{==

Formatting can also be applied to blocks by putting the opening and closing
tags on separate lines and adding new lines between the tags and the content.

==}

高亮文本
- ==This was marked (highlight)==

下划线
- ^^This was inserted (underline)^^

删除线
- ~~This was deleted (strikethrough)~~

上下标
- H~2~O
- A^T^A

键盘键
++ctrl+alt+del++
```

Text can be {--deleted--} and replacement text {++added++}. This can also be
combined into {~~one~>a single~~} operation. {==Highlighting==} is also
possible {>>and comments can be added inline<<}.

{==

Formatting can also be applied to blocks by putting the opening and closing
tags on separate lines and adding new lines between the tags and the content.

==}

- ==This was marked (highlight)==
- ^^This was inserted (underline)^^
- ~~This was deleted (strikethrough)~~
- H~2~O
- A^T^A

++ctrl+alt+del++

## Grids 网格

---

### Configuration

```yaml title="mkdocs.yml"
markdown_extensions: 
  - attr_list
  - md_in_html
```

### Usage

**卡片网格**

列表语法本质上是卡片网格的快捷方式，它由一个无序（或有序）列表组成，该列表由 div 包装，同时具有 grid 和 cards 类：

<div class="grid cards" markdown>

- :fontawesome-brands-html5: __HTML__ for content and structure
- :fontawesome-brands-js: __JavaScript__ for interactivity
- :fontawesome-brands-css3: __CSS__ for text running out of boxes
- :fontawesome-brands-internet-explorer: __Internet Explorer__ ... huh?

</div>

列表元素可以包含任意 Markdown，只要周围的 div 定义了 markdown 属性。下面是一个更复杂的示例，其中包括图标和链接：

<div class="grid cards" markdown>

- :material-clock-fast:{ .lg .middle } __Set up in 5 minutes__

    ---

    Install [`mkdocs-material`](#) with [`pip`](#) and get up
    and running in minutes

    [:octicons-arrow-right-24: Getting started](#)

- :fontawesome-brands-markdown:{ .lg .middle } __It's just Markdown__

    ---

    Focus on your content and generate a responsive and searchable static site

    [:octicons-arrow-right-24: Reference](#)

- :material-format-font:{ .lg .middle } __Made to measure__

    ---

    Change the colors, fonts, language, icons, logo and more with a few lines

    [:octicons-arrow-right-24: Customization](#)

- :material-scale-balance:{ .lg .middle } __Open Source, MIT__

    ---

    Material for MkDocs is licensed under MIT and available on [GitHub]

    [:octicons-arrow-right-24: License](#)

</div>

块语法允许将卡片与其他元素一起排列在网格中，如通用网格部分所述。只需将 card 类添加到网格内的任何 block 元素即可：

<div class="grid" markdown>

:fontawesome-brands-html5: __HTML__ for content and structure
{ .card }

:fontawesome-brands-js: __JavaScript__ for interactivity
{ .card }

:fontawesome-brands-css3: __CSS__ for text running out of boxes
{ .card }

> :fontawesome-brands-internet-explorer: __Internet Explorer__ ... huh?

</div>

使用通用网格

<div class="grid" markdown>

=== "Unordered list"

    * Sed sagittis eleifend rutrum
    * Donec vitae suscipit est
    * Nulla tempor lobortis orci

=== "Ordered list"

    1. Sed sagittis eleifend rutrum
    2. Donec vitae suscipit est
    3. Nulla tempor lobortis orci

``` title="Content tabs"
=== "Unordered list"

    * Sed sagittis eleifend rutrum
    * Donec vitae suscipit est
    * Nulla tempor lobortis orci

=== "Ordered list"

    1. Sed sagittis eleifend rutrum
    2. Donec vitae suscipit est
    3. Nulla tempor lobortis orci
```

</div>

## Icon, Emojis 图标, 表情包

---

Material for MkDocs 的最佳功能之一是可以在您的项目文档中使用超过 10,000 个图标和数千个表情符号，而几乎不需要额外的努力。此外，可以在 mkdocs.yml、文档和模板中添加和使用自定义图标 。

### Configuration

```yaml title="mkdocs.yml"
markdown_extensions:
  - attr_list
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg

extra_css:
  - stylesheets/extra.css
```

### Usage

```text title="Icon, Emojis"
:smile:
:fontawesome-regular-face-laugh-wink:
:fontawesome-brands-youtube:{ .youtube }
:octicons-heart-fill-24:{ .heart }
```

:smile:
:fontawesome-regular-face-laugh-wink:
:fontawesome-brands-youtube:{ .youtube }
:octicons-heart-fill-24:{ .heart }

## Image 图片

---

虽然图像是 Markdown 的一等公民，并且是核心语法的一部分，但使用它们可能很困难。MkDocs 的 Material 使图像处理更加舒适，提供了图像对齐和图像标题的样式。

### Configuration

```yaml title="mkdocs.yml"
markdown_extensions:
  - attr_list
  - md_in_html
  - pymdownx.blocks.caption

plugins:
  - glightbox
```

如果您想在文档中添加图像缩放功能，glightbox 插件是一个很好的选择，因为它与 Material for MkDocs 完美集成。用 pip 安装它：

```bash
pip install mkdocs-glightbox
```

### Usage

```text title="Image"
左对齐，注意align不支持居中对齐
![Image title](https://dummyimage.com/600x400/eee/aaa){ align=left }

图注 Caption
<figure markdown="span">
  ![Image title](https://dummyimage.com/600x400/){ width="300" }
  <figcaption>Image caption</figcaption>
</figure>

图注另外一种写法
![Image title](https://dummyimage.com/600x400/){ width="300" }
/// caption
Image caption
///

懒加载
![Image title](https://dummyimage.com/600x400/){ loading=lazy }

根据明暗主题显示不同图片
![Image title](https://dummyimage.com/600x400/f5f5f5/aaaaaa#only-light)
![Image title](https://dummyimage.com/600x400/21222c/d5d7e2#only-dark)
```

![Image title](https://dummyimage.com/600x400/){ width="300" }
/// caption
Image caption
///

## Lists 列表

---

MkDocs 的 Material 支持多种类型的列表，以满足不同的用例，包括无序列表和有序列表 （通过标准 Markdown 支持），以及定义列表和任务列表 （通过扩展支持）。

### Configuration

```yaml title="mkdocs.yml"
markdown_extensions:
  - def_list
  - pymdownx.tasklist:
      custom_checkbox: true
```

### Usage

```text title="Lists demo"

无序列表
- Nulla et rhoncus turpis. Mauris ultricies elementum leo. Duis efficitur
  accumsan nibh eu mattis. Vivamus tempus velit eros, porttitor placerat nibh
  lacinia sed. Aenean in finibus diam.

    * Duis mollis est eget nibh volutpat, fermentum aliquet dui mollis.
    * Nam vulputate tincidunt fringilla.
    * Nullam dignissim ultrices urna non auctor.

有序列表
1.  Vivamus id mi enim. Integer id turpis sapien. Ut condimentum lobortis
    sagittis. Aliquam purus tellus, faucibus eget urna at, iaculis venenatis
    nulla. Vivamus a pharetra leo.

    1.  Vivamus venenatis porttitor tortor sit amet rutrum. Pellentesque aliquet
        quam enim, eu volutpat urna rutrum a. Nam vehicula nunc mauris, a
        ultricies libero efficitur sed.

    2.  Morbi eget dapibus felis. Vivamus venenatis porttitor tortor sit amet
        rutrum. Pellentesque aliquet quam enim, eu volutpat urna rutrum a.

        1.  Mauris dictum mi lacus
        2.  Ut sit amet placerat ante
        3.  Suspendisse ac eros arcu

定义列表
`Lorem ipsum dolor sit amet`

:   Sed sagittis eleifend rutrum. Donec vitae suscipit est. Nullam tempus
    tellus non sem sollicitudin, quis rutrum leo facilisis.

`Cras arcu libero`

:   Aliquam metus eros, pretium sed nulla venenatis, faucibus auctor ex. Proin
    ut eros sed sapien ullamcorper consequat. Nunc ligula ante.

    Duis mollis est eget nibh volutpat, fermentum aliquet dui mollis.
    Nam vulputate tincidunt fringilla.
    Nullam dignissim ultrices urna non auctor.

任务列表
- [x] Lorem ipsum dolor sit amet, consectetur adipiscing elit
- [ ] Vestibulum convallis sit amet nisi a tincidunt
    * [x] In hac habitasse platea dictumst
    * [x] In scelerisque nibh non dolor mollis congue sed et metus
    * [ ] Praesent sed risus massa
- [ ] Aenean pretium efficitur erat, donec pharetra, ligula non scelerisque
```

`Lorem ipsum dolor sit amet`

:   Sed sagittis eleifend rutrum. Donec vitae suscipit est. Nullam tempus
    tellus non sem sollicitudin, quis rutrum leo facilisis.

`Cras arcu libero`

:   Aliquam metus eros, pretium sed nulla venenatis, faucibus auctor ex. Proin
    ut eros sed sapien ullamcorper consequat. Nunc ligula ante.

    Duis mollis est eget nibh volutpat, fermentum aliquet dui mollis.
    Nam vulputate tincidunt fringilla.
    Nullam dignissim ultrices urna non auctor.

- [x] Lorem ipsum dolor sit amet, consectetur adipiscing elit
- [ ] Vestibulum convallis sit amet nisi a tincidunt
    * [x] In hac habitasse platea dictumst
    * [x] In scelerisque nibh non dolor mollis congue sed et metus
    * [ ] Praesent sed risus massa
- [ ] Aenean pretium efficitur erat, donec pharetra, ligula non scelerisque

## Math 数学

---

MathJax 和 KaTeX 是两个用于在浏览器中显示数学内容的常用库。尽管这两个库提供类似的功能，但它们使用不同的语法并具有不同的配置选项。本文档站点提供了有关如何轻松地将它们与 Material for MkDocs 集成的信息。

- 速度 ： KaTeX 通常比 MathJax 快。如果您的网站需要快速渲染大量复杂方程式，KaTeX 可能是更好的选择。
- 语法支持 ：MathJax 支持更广泛的 LaTeX 命令，并且可以处理各种数学标记语言（如 AsciiMath 和 MathML）。如果您需要高级 LaTeX 功能，MathJax 可能更合适。

### Configuration

=== "docs/javascript/mathjax.js"

    ``` js
    window.MathJax = {
      tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true
      },
      options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
      }
    };

    document$.subscribe(() => { 
      MathJax.startup.output.clearCache()
      MathJax.typesetClear()
      MathJax.texReset()
      MathJax.typesetPromise()
    })
    ```

=== "mkdocs.yml"

    ``` yaml
    markdown_extensions:
      - pymdownx.arithmatex:
          generic: true

    extra_javascript:
      - javascripts/mathjax.js
      - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js
    ```

=== "docs/javascript/katex.js"

    ```js
    document$.subscribe(({ body }) => { 
      renderMathInElement(body, {
        delimiters: [
          { left: "$$",  right: "$$",  display: true },
          { left: "$",   right: "$",   display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true }
        ],
      })
    })
    ```

=== "mkdocs.yml"

    ```yaml
    markdown_extensions:
      - pymdownx.arithmatex:
          generic: true

    extra_javascript:
      - javascripts/katex.js
      - https://unpkg.com/katex@0/dist/katex.min.js
      - https://unpkg.com/katex@0/dist/contrib/auto-render.min.js

    extra_css:
      - https://unpkg.com/katex@0/dist/katex.min.css
    ```

### Usage

```text title="math demo"
$$
\cos x=\sum_{k=0}^{\infty}\frac{(-1)^k}{(2k)!}x^{2k}
$$

The homomorphism $f$ is injective if and only if its kernel is only the
singleton set $e_G$, because otherwise $\exists a,b\in G$ with $a\neq b$ such
that $f(a)=f(b)$.
```

$$
\cos x=\sum_{k=0}^{\infty}\frac{(-1)^k}{(2k)!}x^{2k}
$$

The homomorphism $f$ is injective if and only if its kernel is only the
singleton set $e_G$, because otherwise $\exists a,b\in G$ with $a\neq b$ such
that $f(a)=f(b)$.

## Tooltips

---

技术文档经常会使用许多首字母缩略词，这可能需要额外的解释，特别是对于项目的新用户。对于这些问题，Material for MkDocs 使用 Markdown 扩展的组合来启用站点范围的词汇表。

### Configuration

```yaml title="mkdocs.yml"
# Improved tooltips
theme:
  features:
    - content.tooltips

# tooltips extension
markdown_extensions:
  - abbr
  - attr_list
  - pymdownx.snippets

# 使用 docs 文件夹外部的专用文件时，请将父目录添加到监视文件夹列表中，以便在更新术语表文件时，在运行 mkdocs serve 时自动重新加载项目。
watch:
  - includes
```

### Usage

[Hover me](https://example.com "I'm a tooltip!")

[Hover me][example]

  [example]: https://example.com "I'm a tooltip!"

:material-information-outline:{ title="Important information" }

The HTML specification is maintained by the W3C.

*[HTML]: Hyper Text Markup Language
*[W3C]: World Wide Web Consortium

=== "includes/abbreviations.md"
    ```markdown
    *[HTML]: Hyper Text Markup Language
    *[W3C]: World Wide Web Consortium
    ```
=== "mkdocs.yml"

    ```yaml
    markdown_extensions:
      - pymdownx.snippets:
          auto_append:
            - includes/abbreviations.md
    ```
