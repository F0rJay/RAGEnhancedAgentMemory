#!/usr/bin/env python3
"""
下载嵌入模型脚本

下载并验证嵌入模型，支持：
- 进度显示
- 断点续传
- 模型验证
- 缓存路径显示
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置 HuggingFace 镜像源（如果设置了）
try:
    from src.config import get_settings
    settings = get_settings()
    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint
        print(f"使用 HuggingFace 镜像源: {settings.hf_endpoint}")
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("错误: sentence-transformers 未安装")
    print("请运行: pip install sentence-transformers>=2.3.0")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("提示: tqdm 未安装，将不显示下载进度条")
    print("可选安装: pip install tqdm")


def download_model(model_name: str = "BAAI/bge-large-en-v1.5", verify: bool = True):
    """
    下载嵌入模型
    
    Args:
        model_name: 模型名称或路径
        verify: 是否验证模型
    """
    print("=" * 60)
    print("嵌入模型下载工具")
    print("=" * 60)
    print(f"\n模型名称: {model_name}")
    print(f"下载位置: ~/.cache/huggingface/hub/")
    print("\n开始下载...")
    print("注意: 首次下载可能需要较长时间，请耐心等待")
    print("-" * 60)
    
    try:
        # 下载模型（sentence-transformers 会自动处理下载和缓存）
        print(f"\n正在下载模型: {model_name}")
        print("提示: sentence-transformers 会自动下载并缓存模型")
        print("     如果网络较慢，请耐心等待...")
        
        # SentenceTransformer 会自动显示下载进度（如果 huggingface_hub 支持）
        model = SentenceTransformer(model_name)
        
        print("\n✓ 模型下载完成！")
        
        # 显示模型信息
        print("\n" + "-" * 60)
        print("模型信息:")
        print("-" * 60)
        print(f"模型名称: {model_name}")
        print(f"模型路径: {model._modules['0'].auto_model.config.name_or_path}")
        
        # 测试模型
        if verify:
            print("\n正在验证模型...")
            test_texts = ["Hello world", "测试文本"]
            embeddings = model.encode(test_texts)
            
            print(f"✓ 模型验证成功")
            print(f"  - 测试文本数量: {len(test_texts)}")
            print(f"  - 向量维度: {len(embeddings[0])}")
            print(f"  - 向量类型: {type(embeddings)}")
        
        # 显示缓存路径
        try:
            from huggingface_hub import snapshot_download
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            print(f"\n模型缓存目录: {cache_dir}")
            
            # 查找模型文件
            model_dirs = list(cache_dir.glob(f"models--*"))
            if model_dirs:
                print(f"已缓存的模型数量: {len(model_dirs)}")
        except:
            pass
        
        print("\n" + "=" * 60)
        print("✓ 模型下载和验证完成！")
        print("=" * 60)
        print("\n现在可以在代码中使用该模型了:")
        print(f"  from sentence_transformers import SentenceTransformer")
        print(f"  model = SentenceTransformer('{model_name}')")
        print("\n或者在 .env 文件中设置:")
        print(f"  EMBEDDING_MODEL={model_name}")
        
        return True
        
    except ConnectionError as e:
        print("\n❌ 网络连接错误")
        print(f"错误: {e}")
        print("\n解决方案:")
        print("1. 检查网络连接")
        print("2. 配置 HuggingFace 镜像源")
        print("3. 使用代理")
        return False
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="下载嵌入模型")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="模型名称（默认: BAAI/bge-large-en-v1.5）"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过模型验证"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出推荐的模型"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("推荐的嵌入模型:")
        print("-" * 60)
        models = [
            ("BAAI/bge-large-en-v1.5", "大型英文模型，1024维，性能最好"),
            ("BAAI/bge-base-en-v1.5", "基础英文模型，768维，平衡性能和速度"),
            ("all-MiniLM-L6-v2", "小型通用模型，384维，速度快"),
            ("paraphrase-multilingual-MiniLM-L12-v2", "多语言模型，支持中文"),
        ]
        for model_name, description in models:
            print(f"  {model_name:50} - {description}")
        return
    
    success = download_model(args.model, verify=not args.no_verify)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断下载")
        sys.exit(1)
    except Exception as e:
        print(f"\n未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
