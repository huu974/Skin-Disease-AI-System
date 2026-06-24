#日志记录器，同时输出到文件和终端， 用于获取日志器
import sys, os
import time
import logging
from utils.path_tool import get_abs_path
from datetime import datetime

#分类
class Logger:
    def __init__(self, log_path, log_terminal=True):
        #初始化控制台+文件双重输出
        self.terminal = sys.stdout
        self.log_terminal = log_terminal
        
        # 创建目录
        dir_path = os.path.dirname(log_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        # 打开日志文件
        self.log_file = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.log_file.write(message)
        self.log_file.flush()

        if self.log_terminal:
            self.terminal.write(message)

    def flush(self):
        self.log_file.flush()



#Rag与agent日志调用输出
#日志保存的根目录
LOG_ROOT = get_abs_path("logs")

os.makedirs(LOG_ROOT, exist_ok=True)


#定义默认的日志格式，包含时间，模块名，级别，文件位置，消息内容(lineno: 代码行号       message: 消息)
DEFAULT_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)



#获取日志器的工厂函数
def get_logger(
        name:str='agent',
        console_level:int=logging.INFO,         #控制台日志级别，默认为INFO
        file_level:int=logging.DEBUG,           #文件日志级别，默认为DEBUG
        log_file:str=None,                      #日志文件路径，默认为None时会自动生成
):
    #创建日志器
    logger = logging.getLogger(name)
    #设置日志级别
    logger.setLevel(logging.DEBUG)

    #检查日志器是否已有处理器
    if logger.handlers:
        return logger

    #创建控制台输出处理器(流式输出)
    consloe_handler = logging.StreamHandler()
    #设置控制台处理器的日志级别
    consloe_handler.setLevel(console_level)
    #为控制台处理器设置日志格式
    consloe_handler.setFormatter(DEFAULT_FORMAT)

    #将控制台处理器添加到日志器中
    logger.addHandler(consloe_handler)

    #文件处理器
    if not log_file:        #日志文件存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")

    #创建文件输出处理器
    file_handler = logging.FileHandler(log_file,encoding="utf-8")
    #设置文件处理器的日志级别
    file_handler.setLevel(file_level)
    #为文件处理器设置日志格式
    file_handler.setFormatter(DEFAULT_FORMAT)
    #将文件处理器添加到日志器中
    logger.addHandler(file_handler)

    #返回日志器
    return logger

logger = get_logger()


if __name__ == '__main__':
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")













