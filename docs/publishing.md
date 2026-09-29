# 发布到 GitHub

## 首次发布前

1. 本项目仓库为 `cloud0203/pytorch-training-engineering`。
2. 仓库可见性为 public。
3. 采用根目录 [MIT License](../LICENSE)，README 已标明许可证。
4. 派生或迁移仓库时，同步修改 README 中的克隆地址和本文的远程地址。
5. 运行 `python scripts/validate_skill.py`，确认安装示例、显式调用策略与文件内容一致。

## 创建仓库与上传

在 GitHub 创建同名空仓库。首次推送已有本地项目时，不在网页端预生成 README、LICENSE 或 .gitignore，避免两套初始历史。[GitHub 官方步骤](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

在本项目目录执行下列命令。`git init` 仅在尚未初始化时执行；将用户名和仓库名替换成实际值。使用已有 Git 身份和 GitHub 认证，不把访问令牌写入命令或 remote URL。

```bash
git init -b main
git add README.md LICENSE CONTRIBUTING.md docs skills scripts requirements-dev.txt .github .gitignore
git diff --cached --stat
git commit -m "Initial release of PyTorch training engineering skill"
git remote add origin https://github.com/cloud0203/pytorch-training-engineering.git
git push -u origin main
```

已存在 origin 时先检查 `git remote -v`，不要盲目覆盖。上传内容仅限本独立项目，不在训练项目根目录执行 `git add .`。

## GitHub 页面信息

可使用以下 About 文案：

> A Chinese-language agent skill for building reproducible PyTorch training projects with explicit user approval.

可选 Topics：`agent-skills`、`codex`、`pytorch`、`deep-learning`、`training`、`reproducibility`。

上传后确认 README 正常渲染、内部链接可用、Actions 校验成功，再建立首个版本。当前仓库未放置 CI 成功或许可证徽章，避免尚未验证就宣称通过。

## 版本发布

首个可用版本可使用 `v0.1.0`。完成检查后：

```bash
git tag -a v0.1.0 -m "Initial skill release"
git push origin v0.1.0
```

在 GitHub Releases 中选择该标签，简述能力、确认机制、安装路径和已知限制。保留标签对应的内容，后续变更发布新版本。

这是一套可直接从 GitHub 分发的 skill 文件，不自动注册到插件目录。若以后需要插件市场分发，再按目标平台规范另行打包。
