"""
文件作用：用于获取所传相对路径的绝对路径，将路径更完善
"""

import os


# 获取项目根目录
def get_project_root() -> str:
    #1.获取当前文件的绝对路径
    current_file = os.path.abspath(__file__)
    #2.获取文件所在文件夹的绝对路径
    current_dir = os.path.dirname(current_file)
    #3.获取项目根目录的绝对路径
    project_root = os.path.dirname(current_dir)

    return project_root


#传递相对路径，得到绝对路径
def get_abs_path(relative_path:str) -> str:
    project_root = get_project_root()

    return os.path.join(project_root, relative_path)

if __name__ == '__main__':
    #E:\py项目\Skin diseases\config\model.yaml
    print(get_abs_path("config/model.yaml"))
