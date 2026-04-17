import pandas as pd
from pathlib import Path

ProjectPath = Path(__file__).parent
DataPath = ProjectPath / 'data'
RawData = DataPath / 'raw'
CleanedData = DataPath / 'cleaned'


def KeepTypst(
        Data: pd.DataFrame
) -> pd.DataFrame:
    """
    去除空数据、损坏文件、二进制文件等
    分离输入数据的Typst数据和非Typst数据
    去除无关的列

    参数:
        Data.columns = ['repo', 'file', 'language', 'license', 'content']

    返回值:
        TypstData, No_TypstData
        Data.columns = ['repo', 'language', 'content']
    """
    Data = Data.dropna() # 删除缺失
    Data = Data.drop(columns=["license", "file"]) # 删除许可证列和文件url列
    TypstData = Data[ Data['language'] == 'typst' ] # 选出typst的部分
    No_TypstData = Data[ Data['language'] == 'markdown' ] # 选出非typst的部分

    return TypstData, No_TypstData


def DropSmallContent(
        Data: pd.DataFrame
) -> pd.DataFrame:
    """
    过滤较短的文本
    如果 行数小于8 或 字符数小于80 则去除该数据

    参数:
        Data.columns = ['repo', 'language', 'content']
    
    返回值:
        Data.columns = ['repo', 'language', 'content']
    """
    content = Data['content'].astype(str)

    Data = Data[
        (
            content.str.replace(" ","")
                    .str.replace("\n","")
                    .str.len() >= 80
        )
        & (content.str.count("\n").add(1) >= 8)
    ]

    return Data


def DropDuplicate(
        Data: pd.DataFrame
) -> pd.DataFrame:
    """
    去重

    参数:
        Data.columns = ['repo', 'language', 'content']

    返回值:
        Data.columns = ['repo', 'language', 'content']
    """

    Data = Data.drop_duplicates() # 去除完全重复的行
    Data = Data.drop_duplicates(subset=['repo']) # 去除来自同一个仓库的代码

    return Data


def clean(
        Data: pd.DataFrame,
        local_dir: Path
) -> pd.DataFrame:
    
    TypstData, No_TypstData = KeepTypst(Data)
    TypstData = DropSmallContent(TypstData)
    No_TypstData = DropSmallContent(No_TypstData)
    TypstData = DropDuplicate(TypstData)
    No_TypstData = DropDuplicate(No_TypstData)

    local_dir.mkdir(exist_ok=True)
    TypstData.to_json(local_dir / 'typst.json')
    No_TypstData.to_json(local_dir / 'no_typst.json')

    return TypstData, No_TypstData

if __name__ == '__main__':

    RawTrainData = pd.read_json(RawData / 'train/typst_train.json')
    RawTestData = pd.read_json(RawData/ 'test/typst_test.json')


    # RawTrainData.shape = (21069, 5)
    # RawTestData.shape = (1000, 5)
    # RawData.columns = ['repo', 'file', 'language', 'license', 'content']
    # language of Data has two value: 'typst' 'markdown'
    CleanedData.mkdir(exist_ok=True)
    CleanedTrainData, _ = clean(RawTrainData, CleanedData / 'train') # 训练集的清理
    CleanedTestData, _ = clean(RawTestData, CleanedData / 'test') # 测试集的清理

    # CleanedTrainData.shape = (2580, 3)
    # CleanedTestData.shape = (458, 3)
    # CleanData.columns = ['repo', 'language', 'content']
    print(CleanedTrainData.shape)
    print(CleanedTestData.shape)
    print(CleanedTrainData.columns)

    