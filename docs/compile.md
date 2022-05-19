# 依赖

embedx 需要一个支持 **C++11** 的编译器。

embedx 依赖[deepx_core](https://github.com/Tencent/deepx_core)

## 有网络环境下编译

```shell
# 机器可以访问 github.com, 下面的命令会自动下载依赖 deepx_core
git submodule update --init
# 编译 embedx
make -j8
# 编译出的程序在 build_xxx 目录下
```

## 无网络环境下编译

- 手动下载[deepx_core](https://github.com/Tencent/deepx_core)

- 手动下载[embedx](https://github.com/Tencent/embedx)

- 将 **deepx_core** 放到 **embedx 目录下**

```shell
# 编译 embedx
make -j8
# 编译出的程序在 build_xxx 目录下
```

## 编译优化

参考[deepx 编译优化](https://github.com/Tencent/deepx_core/blob/master/doc/compilation.md)
