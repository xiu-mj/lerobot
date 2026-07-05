# 文档生成指南

版权所有 2020 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0（以下简称「许可证」）授权；
除非遵守许可证，否则你不得使用此文件。
你可以在以下地址获取许可证副本：

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件
按「**原样**」提供，
**不提供任何明示或暗示的担保或条件**。
详见许可证，了解具体的权限和
限制条款。

---

# 文档生成

想要生成文档，首先需要进行构建。构建文档需要多个依赖包，
你可以在代码仓库根目录执行以下命令进行安装：

```bash
pip install -e . -r docs-requirements.txt
```

同时还需要安装 `nodejs`，请参考其[安装页面](https://nodejs.org/en/download)。

---

**注意**

仅当你需要**本地预览文档**时（例如修改内容后想查看效果再提交），才需要生成文档。
**不需要将构建后的文档文件提交到 Git。**

---

## 构建文档

安装好 `doc-builder` 与其他依赖包后，执行以下命令即可生成文档：

```bash
doc-builder build lerobot docs/source/ --build_dir ~/tmp/test-build
```

你可以修改 `--build_dir` 指定任意临时目录。该命令会创建目录并生成 MDX 文件，
这些文件会在官网渲染为文档页面，可使用任意 Markdown 编辑器查看。

## 预览文档

如需本地预览，先安装 `watchdog`：

```bash
pip install watchdog
```

然后运行：

```bash
doc-builder preview lerobot docs/source/
```

预览地址：[http://localhost:3000](http://localhost:3000)。
提交 PR 后也可预览：机器人会自动添加评论，提供包含你修改的文档预览链接。

---

**注意**

`preview` 命令仅对**已存在**的文档文件生效。
新增文件时，需要先更新 `_toctree.yml`，再重启预览命令（`ctrl-c` 停止后重新执行）。

---

## 在导航栏添加新条目

支持文件格式：Markdown（`.md`）。

在 `source` 目录下新建对应文件，然后在 [`_toctree.yml`](https://github.com/huggingface/lerobot/main/docs/source/_toctree.yml) 中
填入**不带后缀**的文件名，即可加入导航目录。

## 重命名章节标题或移动章节

重命名章节、移动章节时，建议保留旧链接可用。
因为旧链接可能已被 Issue、论坛、社交媒体引用，保留跳转能大幅提升用户体验。

做法：在原章节所在文档末尾保留一个**跳转映射**，关键是**保留原锚点**。

例如：章节从「Section A」改名为「Section B」，在文件末尾添加：

```
已移动的章节：

[ <a href="#section-b">Section A</a><a id="section-a"></a> ]
```

如果移动到其他文件：

```
已移动的章节：

[ <a href="../new-file#section-b">Section A</a><a id="section-a"></a> ]
```

使用**相对路径**链接，保证多版本文档正常工作。

完整示例可参考：[transformers Trainer 文档末尾](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/trainer.md)。

### 添加新教程

新增教程或章节分两步：

1. 在 `./source` 下新建文件，格式可为 ReStructuredText（`.rst`）或 Markdown（`.md`）。
2. 在 `./source/_toctree.yml` 的对应位置添加该文件路径。

请放在合适的章节下，如有疑问可在 GitHub Issue 或 PR 中询问。

### 编写文档内容

需要以代码格式显示的内容用反引号包裹：\`like so\`。
参数名、`True`、`None`、字符串等通常都需要用代码格式标注。

#### 多行代码块

多行代码块使用 Markdown 标准的三个反引号：

````
```
# 第一行代码
# 第二行代码
# 以此类推
```
````

#### 插入图片

仓库体积增长较快，**禁止直接添加大体积文件**（图片、视频等非文本文件）。
推荐上传到 Hugging Face 托管的数据集，通过 URL 引用。

建议使用官方数据集：[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)。

外部贡献者可先将图片放到 PR 里，再请 Hugging Face 成员迁移至该数据集。

---

### 术语简明对照
- **doc-builder**: Hugging Face 专用文档构建工具
- **PR (Pull Request)**: 代码合并请求
- **MDX**: 支持 JSX 组件的增强版 Markdown
- **toctree**: 文档目录树（导航结构）
- **anchor**: 页面内锚点（# 后面的部分）