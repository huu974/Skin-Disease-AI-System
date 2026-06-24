"""
命令行参数解析
"""

import argparse
import yaml
import sys



def parse():
    #创建参数解析器
    parser = argparse.ArgumentParser(
        description="Skin diseases image classification"
    )

    #模型选择
    parser.add_argument('--model',default='efficientnet_b3',type=str,help='模型名称')

    #命令行参数列表
    #参1：参数名称  参2：参数的默认值    参3：参数类型     参4：参数的描述
    parser.add_argument('--config',default='instructions.yml',type=str,help='yaml配置文件名称')

    #一般参数
    parser.add_argument('--datapath-train',default='./skin diseases/train-new',type=str,help='训练数据集路径')
    parser.add_argument('--val', default=False, type=bool, help='是否验证')
    parser.add_argument('--datapath-val', default='./skin diseases/val', type=str, help='验证数据集路径')
    parser.add_argument('--datapath-test', default='./skin diseases/test', type=str, help='测试数据集路径')
    parser.add_argument('--batch-size',default=16,type=int,help='批量大小')
    parser.add_argument('--channels-last',default=True,type=bool,help='是否使用channels_last内存格式')
    parser.add_argument('--save-path',default='./variables',type=str,help='保存变量的路径')

    #优化器
    parser.add_argument('--weight-decay', '--wd',default=1e-3, type=float, help='权重衰减 (默认: 1e-4)')
    parser.add_argument('--optimizer',default='Adam',type=str,help='优化器类型')
    parser.add_argument('--epochs',type=int,default=100,help='总训练轮数')

    #学习率
    parser.add_argument('--lr',default=1e-3,type=float,help='初始学习率')

    #日志配置
    parser.add_argument('--logterminal', default=True, type=bool, help='是否打印日志到终端')
    parser.add_argument('--resume',default='',type=str,help='从checkpoint中恢复训练的路径')



    args = parser.parse_args()



    #获取yaml文件内容
    with open('config/default.yml','r',encoding='utf-8') as f:
        yaml_dict = yaml.load(f,Loader=yaml.FullLoader)

    #先用yaml配置更新args
    args.__dict__.update(yaml_dict)

    #命令行参数覆盖yaml配置
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--'):
            # 解析参数名
            if '=' in arg:
                key, value = arg.split('=', 1)
            else:
                key = arg
                value = sys.argv[i + 1] if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--') else True
            
            # 转换为 args 属性名
            key = key.replace('--', '').replace('-', '_')
            
            # 更新 args
            if key in args.__dict__:
                # 类型转换
                original_type = type(yaml_dict.get(key, args.__dict__.get(key)))
                try:
                    args.__dict__[key] = original_type(value)
                except:
                    args.__dict__[key] = value

    return args




























