import os
import hashlib
from pathlib import Path

def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"读取文件出错 {file_path}: {e}")
        return None

def get_all_files(directory):
    """获取目录下所有文件的相对路径"""
    files = {}
    base_path = Path(directory)
    for file_path in base_path.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(base_path)
            files[str(relative_path)] = str(file_path)
    return files

def compare_directories(dir_a, dir_b):
    """比较两个目录"""
    print(f"正在扫描目录A: {dir_a}")
    files_a = get_all_files(dir_a)
    
    print(f"正在扫描目录B: {dir_b}")
    files_b = get_all_files(dir_b)
    
    # (1) A中有但B中没有的文件
    only_in_a = set(files_a.keys()) - set(files_b.keys())
    
    # (2) 同名但内容不同的文件
    common_files = set(files_a.keys()) & set(files_b.keys())
    different_content = []
    
    print("\n正在比较同名文件内容...")
    for relative_path in common_files:
        hash_a = calculate_file_hash(files_a[relative_path])
        hash_b = calculate_file_hash(files_b[relative_path])
        
        if hash_a and hash_b and hash_a != hash_b:
            different_content.append(relative_path)
    
    # 输出结果
    print("\n" + "="*60)
    print("扫描结果:")
    print("="*60)
    
    print(f"\n(1) A中有但B中没有的文件 (共 {len(only_in_a)} 个):")
    if only_in_a:
        for file in sorted(only_in_a):
            print(f"  - {file}")
    else:
        print("  无")
    
    print(f"\n(2) 同名但内容不同的文件 (共 {len(different_content)} 个):")
    if different_content:
        for file in sorted(different_content):
            print(f"  - {file}")
            print(f"    A: {files_a[file]}")
            print(f"    B: {files_b[file]}")
    else:
        print("  无")
    
    # 返回结果供进一步处理
    return {
        'only_in_a': list(only_in_a),
        'different_content': different_content
    }

if __name__ == "__main__":
    # 设置要比较的两个目录
    dir_a = input("请输入目录A的路径: ").strip()
    dir_b = input("请输入目录B的路径: ").strip()
    
    # 检查目录是否存在
    if not os.path.exists(dir_a):
        print(f"错误: 目录A不存在: {dir_a}")
        exit(1)
    if not os.path.exists(dir_b):
        print(f"错误: 目录B不存在: {dir_b}")
        exit(1)
    
    # 执行比较
    results = compare_directories(dir_a, dir_b)
    
    # 可选: 将结果保存到文件
    save = input("\n是否将结果保存到文件? (y/n): ").strip().lower()
    if save == 'y':
        with open('comparison_result.txt', 'w', encoding='utf-8') as f:
            f.write("(1) A中有但B中没有的文件:\n")
            for file in sorted(results['only_in_a']):
                f.write(f"  - {file}\n")
            
            f.write("\n(2) 同名但内容不同的文件:\n")
            for file in sorted(results['different_content']):
                f.write(f"  - {file}\n")
        
        print("结果已保存到 comparison_result.txt")