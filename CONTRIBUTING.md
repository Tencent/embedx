# Contributing

[TOC]

我们提倡通过提 issue 和 pull request 方式来促进 embedx 的发展。

## Issue 提交

请描述清楚使用过程中遇到的问题，提交 issue。

## Pull request

pull request 需要满足 代码要求 和 commit message 要求。

### 代码要求

- 规范

  - `C++` 基本使用 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

- 格式化

  - `C++` 使用 `clang-format 8.0.0` 格式化代码，配置文件是[".clang-format"](.clang-format)

- 检查

  - `C++` 使用 `clang-tidy 8.0.0` 检查代码，配置文件是[".clang-tidy"](.clang-tidy)
  - `C++` 使用 `cpplint` 检查代码，配置文件是["CPPLINT.cfg"](CPPLINT.cfg)

- 提交

  - `C++`，在 commit 提交前运行以下命令进行代码检查

    ```shell
    make -j8 lint
    ```

### commit message 要求

commit message 只使用 ASCII 字符且不超过 50 个 ASCII 字符，结尾不加标点符号。

commit message 的格式是。

```shell
type: subject
type(scope): subject
```

- type（commit 类型）

  - feat: 功能
  - fix: 修复 bug
  - doc: 文档
  - style: 风格/格式
  - refactor: 重构
  - perf: 优化
  - test: 测试
  - build: 构建
  - tool: 工具
  - revert: 回滚
  - misc: 杂项

- scope

  - commit 范围

- subject

  - commit 主题

## 更多问题请联系以下开发者

- yuanhang.nju@gmail.com
- chengchuancoder@gmail.com
- chunchen.scut@gmail.com
- shutingnjupt@gmail.com
- zhouyongnju@gmail.com
- Lthong.brian@gmail.com
