# Git 版本管理与多端同步

## 一次性初始化

本目录已执行 `git init` 后，可在根目录运行：

```powershell
git status
git add pyproject.toml requirements.txt .gitignore README.md configs docs scripts src tests
git commit -m "feat: bootstrap time-dependent PCSEL coupled-wave solver"
```

如果 Git 提示身份未设置，只设置你自己的真实信息：

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

## 连接 GitHub/GitLab/Gitee

先在平台网页上新建一个**空仓库**，不要额外生成 README。然后：

```powershell
git branch -M main
git remote add origin <你的仓库SSH或HTTPS地址>
git push -u origin main
```

推荐 SSH。每台电脑只需配置一次 SSH key；私钥、令牌和 `.env` 永远不要提交。

## 每次工作

开始前：

```powershell
git switch main
git pull --ff-only
git switch -c feature/简短功能名
```

完成一个小而可验证的改动后：

```powershell
pytest
git status
git diff
git add <明确的文件名>
git commit -m "feat: 描述新增功能"
git push -u origin feature/简短功能名
```

在网页发起 Merge Request/Pull Request，检查后合并到 `main`。实验性改动不要直接堆到 `main`。

## 多电脑同步

新电脑首次获取：

```powershell
git clone <仓库地址> PCSELSim
cd PCSELSim
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

平时遵守“先拉取、后修改、再推送”。如果两台电脑都改了同一文件，先提交本机修改，再运行
`git pull --rebase`，解决冲突并测试后推送。不要用网盘同步 `.git` 目录；Git 远端就是代码同步源。

## 仿真数据策略

`results/`、`.npz` 和缓存默认忽略，因为大二进制数据会迅速膨胀仓库。建议：

- 小型基准 CSV/PNG：可放在 `tests/reference/` 并正常提交。
- 大型结果：放实验数据盘、对象存储或 Git LFS。
- 每个结果目录保存所用 YAML、Git commit hash 和随机种子，以保证可追溯。
- 论文发布节点打标签：`git tag -a v0.1-inoue2019 -m "Inoue 2019 calibrated baseline"`。

