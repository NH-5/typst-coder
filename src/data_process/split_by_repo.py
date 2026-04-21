import pandas as pd
from pathlib import Path

ProjectPath = Path(__file__).resolve().parent.parent.parent
DataPath = ProjectPath / 'data'
CleanedData = DataPath / 'cleaned'
SplitData = DataPath / 'split_by_repo'


def GetRepoStats(
        Data: pd.DataFrame
) -> pd.DataFrame:
    """
    统计每个 repo 的样本数和字符数

    参数:
        Data.columns = ['repo', 'language', 'content']

    返回值:
        RepoStats.columns = ['repo', 'file_count', 'char_count']
    """
    RepoStats = (
        # assign(...) 相当于先临时新增一列 `char_count`
        # 这里统计每一行 content 的字符数，后面要按 repo 汇总
        Data.assign(char_count=Data['content'].astype(str).str.len())
            # groupby('repo') 表示按 repo 分组
            .groupby('repo', as_index=False)
            # agg(...) 表示对每个 repo 做聚合统计
            # file_count: 该 repo 下有多少条样本
            # char_count: 该 repo 下所有 content 的总字符数
            .agg(
                file_count=('content', 'size'),
                char_count=('char_count', 'sum')
            )
    )

    return RepoStats


def AssignRepos(
        RepoStats: pd.DataFrame,
        train_ratio: float = 0.85,
        dev_ratio: float = 0.10,
        test_ratio: float = 0.05,
        random_state: int = 42
) -> dict[str, list[str]]:
    """
    按 repo 进行划分，并优先按总字符数平衡各个 split

    参数:
        RepoStats.columns = ['repo', 'file_count', 'char_count']

    返回值:
        RepoGroups = {
            'train_repo': [...],
            'dev_repo': [...],
            'test_repo': [...]
        }
    """
    total_ratio = train_ratio + dev_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError('train_ratio + dev_ratio + test_ratio 必须大于 0')

    # 把比例归一化，避免传入的比例和不等于 1 时影响结果
    train_ratio = train_ratio / total_ratio
    dev_ratio = dev_ratio / total_ratio
    test_ratio = test_ratio / total_ratio

    # sample(frac=1) 用来打乱 repo 顺序
    # random_state 固定后，每次运行的划分结果都可复现
    RepoStats = RepoStats.sample(frac=1, random_state=random_state)

    # 再按 char_count 从大到小排序，优先处理大 repo
    # 这样做更容易把总字符数分配得比较平衡
    RepoStats = RepoStats.sort_values(
        by='char_count',
        ascending=False,
        kind='stable'
    ).reset_index(drop=True)

    total_chars = RepoStats['char_count'].sum()
    target_chars = {
        'train_repo': total_chars * train_ratio,
        'dev_repo': total_chars * dev_ratio,
        'test_repo': total_chars * test_ratio,
    }
    current_chars = {
        'train_repo': 0,
        'dev_repo': 0,
        'test_repo': 0,
    }
    RepoGroups = {
        'train_repo': [],
        'dev_repo': [],
        'test_repo': [],
    }

    split_names = ['train_repo', 'dev_repo', 'test_repo']

    for row in RepoStats.itertuples(index=False):
        # itertuples(...) 会把每一行变成可读属性的元组对象
        # 例如这里可以用 row.repo / row.char_count 访问字段
        best_split = None
        best_score = None

        for split_name in split_names:
            # projected_chars 表示“如果把当前 repo 放进这个 split 之后”
            # 各个 split 的字符数会变成什么样
            projected_chars = current_chars.copy()
            projected_chars[split_name] += row.char_count

            # score 越小越好
            # 这里比较的是：分配之后，三个 split 离各自目标字符数还有多远
            score = sum(
                abs(projected_chars[name] - target_chars[name])
                for name in split_names
            )

            # tie_breaker 用于在 score 一样时继续比较
            # 第一项优先给“当前更缺字符数”的 split
            # 第二项优先给“当前 repo 数更少”的 split
            tie_breaker = (
                - (target_chars[split_name] - current_chars[split_name]),
                len(RepoGroups[split_name]),
            )

            score = (score, *tie_breaker)

            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        RepoGroups[best_split].append(row.repo)
        current_chars[best_split] += row.char_count

    return RepoGroups


def SplitDataByRepo(
        Data: pd.DataFrame,
        train_ratio: float = 0.85,
        dev_ratio: float = 0.10,
        test_ratio: float = 0.05,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按 repo 划分数据集

    参数:
        Data.columns = ['repo', 'language', 'content']

    返回值:
        TrainRepoData, DevRepoData, TestRepoData
        Data.columns = ['repo', 'language', 'content']
    """
    RepoStats = GetRepoStats(Data)
    RepoGroups = AssignRepos(
        RepoStats,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # isin(...) 表示“repo 是否属于某个 repo 列表”
    # 这里据此把原始 Data 拆成 train/dev/test 三个子集
    TrainRepoData = Data[Data['repo'].isin(RepoGroups['train_repo'])]
    DevRepoData = Data[Data['repo'].isin(RepoGroups['dev_repo'])]
    TestRepoData = Data[Data['repo'].isin(RepoGroups['test_repo'])]

    return TrainRepoData, DevRepoData, TestRepoData


def SaveSplitData(
        TrainRepoData: pd.DataFrame,
        DevRepoData: pd.DataFrame,
        TestRepoData: pd.DataFrame,
        local_dir: Path
) -> None:
    """
    保存 repo-level 划分结果

    参数:
        TrainRepoData, DevRepoData, TestRepoData
        Data.columns = ['repo', 'language', 'content']
    """
    # parents=True: 父目录不存在时一并创建
    # exist_ok=True: 目录已存在时不报错
    (local_dir / 'train_repo').mkdir(parents=True, exist_ok=True)
    (local_dir / 'dev_repo').mkdir(parents=True, exist_ok=True)
    (local_dir / 'test_repo').mkdir(parents=True, exist_ok=True)

    TrainRepoData.to_json(local_dir / 'train_repo/typst.json')
    DevRepoData.to_json(local_dir / 'dev_repo/typst.json')
    TestRepoData.to_json(local_dir / 'test_repo/typst.json')


def PrintSplitStats(
        TrainRepoData: pd.DataFrame,
        DevRepoData: pd.DataFrame,
        TestRepoData: pd.DataFrame
) -> None:
    """
    打印划分后的基础统计信息
    """
    for split_name, split_data in [
        ('train_repo', TrainRepoData),
        ('dev_repo', DevRepoData),
        ('test_repo', TestRepoData),
    ]:
        # nunique() 统计唯一 repo 数量
        char_count = split_data['content'].astype(str).str.len().sum()
        repo_count = split_data['repo'].nunique()
        print(split_name)
        print(split_data.shape)
        print('repo_count =', repo_count)
        print('char_count =', char_count)


if __name__ == '__main__':

    TypstData = pd.read_json(CleanedData / 'train/typst.json')

    TrainRepoData, DevRepoData, TestRepoData = SplitDataByRepo(TypstData)
    SaveSplitData(TrainRepoData, DevRepoData, TestRepoData, SplitData)
    PrintSplitStats(TrainRepoData, DevRepoData, TestRepoData)
