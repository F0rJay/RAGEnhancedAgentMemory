"""
存储效率优化测试脚本（性能基准测试）

运行真实的存储效率优化测试，包括：
1. 压力测试场景：高频重复查询（预期降低率 90%+）
2. 真实场景：正常业务咨询（预期降低率 40-50%）
3. 生产环境模拟：退货流程咨询（预期降低率约 45%）

使用方法：
    python scripts/benchmark/storage_optimization_test.py
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.memory.long_term import LongTermMemory
    from src.config import get_settings
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖：pip install -r requirements.txt")
    sys.exit(1)

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================================
# 测试数据生成
# ============================================================================

def generate_stress_test_dataset() -> List[Dict[str, str]]:
    """
    场景 A：压力测试 - 高频重复询问
    
    模拟用户不断重复询问同一个问题（如："怎么退货？"问了 10 遍）
    预期：存储降低率 90%+
    """
    dataset = []
    
    # 同一问题重复 10 次
    for i in range(10):
        dataset.append({
            "role": "user",
            "content": "怎么退货？"
        })
        dataset.append({
            "role": "assistant",
            "content": "您可以进入订单详情页点击\"申请售后\"来办理退货。"
        })
    
    # 添加一些低价值寒暄
    dataset.extend([
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好！"},
        {"role": "user", "content": "谢谢"},
        {"role": "assistant", "content": "不客气"},
    ])
    
    return dataset


def generate_production_simulation_dataset() -> List[Dict[str, str]]:
    """
    场景 C：生产环境模拟 - 退货流程咨询
    
    模拟真实用户在咨询"退货流程"时的真实表现：
    - 包含寒暄、追问、以及对同一问题的不同侧面描述
    - 语义上存在"轻微重叠"而非"完全重复"
    - 引入"知识点位移"：看似相似但核心细节不同的问题
    - 增加跨主题内容以模拟真实生产环境
    """
    dataset = [
        # 第一轮退货咨询（红色上衣）
        {"role": "user", "content": "你好，在吗？"},
        {"role": "assistant", "content": "您好！请问有什么可以帮您？"},
        {"role": "user", "content": "我上周买的红色上衣尺码不对，怎么退货？"},
        {"role": "assistant", "content": "您可以进入订单详情页点击\"申请售后\"来办理退货。"},
        {"role": "user", "content": "好的"},
        {"role": "assistant", "content": "收到"},
        
        # 第二轮退货咨询（不同商品：蓝色裤子）
        {"role": "user", "content": "我想退掉那个蓝色的裤子。"},
        {"role": "assistant", "content": "可以的，您在订单详情页申请退货即可。"},
        {"role": "user", "content": "那退货流程具体是怎么样的呀？"},
        {"role": "assistant", "content": "填写申请后，快递员会上门取件，仓库收到货后会为您退款。"},
        {"role": "user", "content": "嗯"},
        {"role": "assistant", "content": "明白"},
        
        # 第三轮：语义相似但细节不同（不同退货原因）
        {"role": "user", "content": "我想问下，如果我要退货的话该怎么操作？"},
        {"role": "assistant", "content": "可以在订单详情页申请退货。"},
        {"role": "user", "content": "就是那个退换货的步骤能再说下吗？"},
        {"role": "assistant", "content": "填写申请 → 上门取件 → 仓库验收 → 退款完成。"},
        
        # 跨主题：会员咨询
        {"role": "user", "content": "会员有什么优惠？"},
        {"role": "assistant", "content": "会员享受 9.5 折优惠，生日月额外优惠券。"},
        {"role": "user", "content": "明白了"},
        {"role": "assistant", "content": "好的"},
        
        # 回到退货主题（订单号查询）
        {"role": "user", "content": "我这单的订单号是12345，麻烦查下状态。"},
        {"role": "assistant", "content": "订单号 12345 的退货申请正在处理中，预计 3-5 个工作日完成退款。"},
        {"role": "user", "content": "谢谢"},
        {"role": "assistant", "content": "不客气"},
    ]
    
    return dataset


def generate_real_scenario_dataset() -> List[Dict[str, str]]:
    """
    场景 B：真实场景 - 正常业务咨询（50轮）
    
    模拟 50 轮正常的业务咨询，包含：
    - 少量寒暄和对同一问题的不同表述
    - 多种业务场景（购物、物流、售后等）
    """
    dataset = []
    
    # 业务场景 1：购物咨询（10轮）
    shopping_conversations = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好！有什么可以帮助您的？"},
        {"role": "user", "content": "我想买一件T恤"},
        {"role": "assistant", "content": "好的，请问您需要什么尺码和颜色？"},
        {"role": "user", "content": "M码，白色"},
        {"role": "assistant", "content": "M码白色T恤目前有货，价格是 99 元。"},
        {"role": "user", "content": "包邮吗？"},
        {"role": "assistant", "content": "满 88 元包邮，您的订单可以享受包邮服务。"},
        {"role": "user", "content": "好的，下单"},
        {"role": "assistant", "content": "已为您下单，订单号是 ORDER001。"},
    ]
    dataset.extend(shopping_conversations)
    
    # 业务场景 2：物流查询（10轮，包含重复询问）
    logistics_conversations = [
        {"role": "user", "content": "我的订单什么时候到？"},
        {"role": "assistant", "content": "您的订单预计明天送达。"},
        {"role": "user", "content": "能查下物流信息吗？"},
        {"role": "assistant", "content": "订单 ORDER001 正在派送中，快递员电话：13800138000。"},
        {"role": "user", "content": "物流到哪里了？"},
        {"role": "assistant", "content": "包裹已到达您所在城市的配送站，今天下午会安排派送。"},
        {"role": "user", "content": "谢谢"},
        {"role": "assistant", "content": "不客气，有问题随时联系。"},
    ]
    dataset.extend(logistics_conversations)
    
    # 业务场景 3：退货咨询（15轮，包含语义相似问题）
    return_conversations = [
        {"role": "user", "content": "衣服不合适，想退货"},
        {"role": "assistant", "content": "可以的，您可以在订单详情页申请退货。"},
        {"role": "user", "content": "怎么申请退货？"},
        {"role": "assistant", "content": "进入订单详情，点击\"申请售后\"按钮，选择退货原因。"},
        {"role": "user", "content": "退货流程是什么？"},
        {"role": "assistant", "content": "填写退货申请后，我们会上门取件，收到货后 3-5 个工作日退款。"},
        {"role": "user", "content": "好的"},
        {"role": "user", "content": "我想问下退货怎么操作"},
        {"role": "assistant", "content": "同上，可以在订单详情页申请退货。"},
        {"role": "user", "content": "退换货的步骤能说下吗？"},
        {"role": "assistant", "content": "填写申请 → 上门取件 → 仓库验收 → 退款完成。"},
        {"role": "user", "content": "明白了"},
        {"role": "assistant", "content": "好的，有其他问题随时联系。"},
    ]
    dataset.extend(return_conversations)
    
    # 业务场景 4：其他咨询（补充到50轮）
    other_conversations = [
        {"role": "user", "content": "你们的客服电话是多少？"},
        {"role": "assistant", "content": "客服电话：400-123-4567，工作时间 9:00-18:00。"},
        {"role": "user", "content": "谢谢"},
        {"role": "assistant", "content": "不客气。"},
        {"role": "user", "content": "会员有什么优惠？"},
        {"role": "assistant", "content": "会员享受 9.5 折优惠，生日月额外优惠券。"},
        {"role": "user", "content": "怎么成为会员？"},
        {"role": "assistant", "content": "单笔订单满 299 元即可成为会员。"},
    ]
    
    # 补充低价值寒暄以达到50轮
    filler_count = 50 - len(dataset) - len(other_conversations)
    fillers = [
        {"role": "user", "content": "好的"},
        {"role": "assistant", "content": "收到"},
        {"role": "user", "content": "嗯"},
        {"role": "assistant", "content": "明白"},
    ]
    
    for i in range(filler_count):
        dataset.append(fillers[i % len(fillers)])
    
    dataset.extend(other_conversations)
    
    return dataset[:50]  # 确保正好50轮


# ============================================================================
# 测试工具函数
# ============================================================================

def convert_to_conversation_turns(dataset: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """将角色对话转换为对话轮次格式"""
    turns = []
    current_turn = {}
    
    for item in dataset:
        role = item.get("role", "user")
        content = item.get("content", "")
        
        if role == "user":
            current_turn = {"human": content}
        elif role == "assistant":
            current_turn["ai"] = content
            current_turn["timestamp"] = time.time()
            turns.append(current_turn)
            current_turn = {}
    
    return turns


def extract_key_information(dataset: List[Dict[str, str]]) -> Dict[str, Any]:
    """提取关键信息用于验证知识保留度"""
    key_info = {
        "order_numbers": [],
        "intents": [],
        "entities": [],
    }
    
    for item in dataset:
        content = item.get("content", "")
        # 提取订单号（简单模式匹配）
        if "订单号" in content or "ORDER" in content.upper():
            # 尝试提取订单号
            import re
            order_match = re.search(r'(ORDER\d+|订单号[：:]\s*(\d+))', content, re.IGNORECASE)
            if order_match:
                order_num = order_match.group(1) if order_match.group(1) else order_match.group(2)
                key_info["order_numbers"].append(order_num)
        
        # 提取意图关键词
        if "退货" in content:
            key_info["intents"].append("return")
        if "物流" in content or "配送" in content:
            key_info["intents"].append("logistics")
        if "会员" in content:
            key_info["intents"].append("membership")
    
    return key_info


def check_knowledge_retention(
    long_term_memory: LongTermMemory,
    key_info: Dict[str, Any],
    session_id: str,
) -> Dict[str, Any]:
    """
    检查知识保留度
    
    验证关键信息是否仍然存在于长期记忆中
    """
    retention_results = {
        "order_number_found": False,
        "return_intent_found": False,
        "retention_rate": 0.0,
    }
    
    # 检查订单号
    if key_info["order_numbers"]:
        for order_num in key_info["order_numbers"]:
            results = long_term_memory.search(
                query=f"订单号 {order_num}",
                top_k=5,
                session_id=session_id,
            )
            if results and any(order_num in r.content for r in results):
                retention_results["order_number_found"] = True
                break
    
    # 检查退货意图
    if "return" in key_info["intents"]:
        results = long_term_memory.search(
            query="退货流程",
            top_k=5,
            session_id=session_id,
        )
        if results and any("退货" in r.content for r in results):
            retention_results["return_intent_found"] = True
    
    # 计算保留率
    total_key_info_count = len(key_info["order_numbers"]) + len(key_info.get("intents", []))
    found_count = sum([
        retention_results["order_number_found"],
        retention_results["return_intent_found"],
    ])
    if total_key_info_count > 0:
        retention_results["retention_rate"] = found_count / total_key_info_count
    
    return retention_results


# ============================================================================
# 测试场景运行
# ============================================================================

def run_storage_test(
    dataset: List[Dict[str, str]],
    scenario_name: str,
    collection_prefix: str,
    enable_optimization: bool = True,
) -> Dict[str, Any]:
    """
    运行存储测试
    
    Args:
        dataset: 测试数据集
        scenario_name: 场景名称
        collection_prefix: Collection 名称前缀
        enable_optimization: 是否启用优化
    
    Returns:
        测试结果
    """
    import os
    
    collection_name = f"{collection_prefix}_{scenario_name}_{'optimized' if enable_optimization else 'baseline'}"
    
    # 提取关键信息
    key_info = extract_key_information(dataset)
    
    # 创建长期记忆实例
    long_term_memory = LongTermMemory(
        vector_db="qdrant",
        collection_name=collection_name,
    )
    
    # 获取归档前的存储数
    stats_before = long_term_memory.get_stats()
    count_before = stats_before.get("total_memories", 0)
    if isinstance(count_before, str):
        count_before = 0
    
    # 转换为对话轮次（保持原始数据，让模块自己处理过滤）
    conversation_turns = convert_to_conversation_turns(dataset)
    
    # 归档到长期记忆（让模块内部处理过滤逻辑）
    settings = get_settings()
    archived_ids = long_term_memory.archive_from_short_term(
        conversation_turns=conversation_turns,  # 传入原始数据
        session_id=f"test_{scenario_name}",
        deduplicate=True,
        semantic_dedup=enable_optimization,  # 语义去重要启用优化时才开启
        filter_low_value=enable_optimization and settings.enable_low_value_filter,  # 低价值过滤
    )
    
    # 获取归档后的存储数
    stats_after = long_term_memory.get_stats()
    count_after = stats_after.get("total_memories", 0)
    if isinstance(count_after, str):
        count_after = 0
    
    # 计算实际新增数量
    actual_added = count_after - count_before
    
    # 检查知识保留度
    retention_results = check_knowledge_retention(
        long_term_memory=long_term_memory,
        key_info=key_info,
        session_id=f"test_{scenario_name}",
    )
    
    return {
        "scenario_name": scenario_name,
        "input_count": len(dataset),
        "conversation_turns": len(conversation_turns),
        "filtered_count": len(conversation_turns) if enable_optimization else 0,
        "archived_count": len(archived_ids),
        "count_before": count_before,
        "count_after": count_after,
        "actual_added": actual_added,
        "key_info": key_info,
        "knowledge_retention": retention_results,
    }


# ============================================================================
# 主测试流程
# ============================================================================

def main():
    """主函数"""
    print("=" * 80)
    print("存储效率优化测试 - 性能基准测试")
    print("=" * 80)
    print()
    
    output_dir = Path(__file__).parent.parent / "benchmark_results" / "storage_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 场景 A：压力测试
    print("\n" + "=" * 80)
    print("场景 A：压力测试 - 高频重复询问")
    print("=" * 80)
    print("预期：存储降低率 90%+")
    print()
    
    stress_dataset = generate_stress_test_dataset()
    print(f"生成压力测试数据集: {len(stress_dataset)} 条对话")
    
    baseline_stress = run_storage_test(
        dataset=stress_dataset,
        scenario_name="stress_test",
        collection_prefix="storage_opt",
        enable_optimization=False,
    )
    
    optimized_stress = run_storage_test(
        dataset=stress_dataset,
        scenario_name="stress_test",
        collection_prefix="storage_opt",
        enable_optimization=True,
    )
    
    all_results["stress_test"] = {
        "baseline": baseline_stress,
        "optimized": optimized_stress,
    }
    
    # 场景 B：真实场景
    print("\n" + "=" * 80)
    print("场景 B：真实场景 - 正常业务咨询（50轮）")
    print("=" * 80)
    print("预期：存储降低率 40-50%")
    print()
    
    real_dataset = generate_real_scenario_dataset()
    print(f"生成真实场景数据集: {len(real_dataset)} 条对话")
    
    baseline_real = run_storage_test(
        dataset=real_dataset,
        scenario_name="real_scenario",
        collection_prefix="storage_opt",
        enable_optimization=False,
    )
    
    optimized_real = run_storage_test(
        dataset=real_dataset,
        scenario_name="real_scenario",
        collection_prefix="storage_opt",
        enable_optimization=True,
    )
    
    all_results["real_scenario"] = {
        "baseline": baseline_real,
        "optimized": optimized_real,
    }
    
    # 场景 C：生产环境模拟
    print("\n" + "=" * 80)
    print("场景 C：生产环境模拟 - 退货流程咨询")
    print("=" * 80)
    print("预期：存储降低率约 45%")
    print()
    
    production_dataset = generate_production_simulation_dataset()
    print(f"生成生产环境模拟数据集: {len(production_dataset)} 条对话")
    
    baseline_production = run_storage_test(
        dataset=production_dataset,
        scenario_name="production_simulation",
        collection_prefix="storage_opt",
        enable_optimization=False,
    )
    
    optimized_production = run_storage_test(
        dataset=production_dataset,
        scenario_name="production_simulation",
        collection_prefix="storage_opt",
        enable_optimization=True,
    )
    
    all_results["production_simulation"] = {
        "baseline": baseline_production,
        "optimized": optimized_production,
    }
    
    # 生成报告
    generate_report(all_results, output_dir)
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)


def generate_report(results: Dict[str, Dict[str, Dict[str, Any]]], output_dir: Path):
    """生成测试报告"""
    
    report_path = output_dir / "storage_optimization_report.json"
    
    # 计算各场景的存储降低率
    report_data = {
        "timestamp": time.time(),
        "test_type": "存储效率优化性能基准测试",
        "scenarios": {},
    }
    
    print("\n" + "=" * 80)
    print("测试报告")
    print("=" * 80)
    
    for scenario_name, scenario_results in results.items():
        baseline = scenario_results["baseline"]
        optimized = scenario_results["optimized"]
        
        # 计算存储降低率
        baseline_storage = baseline["actual_added"]
        optimized_storage = optimized["actual_added"]
        
        if baseline_storage > 0:
            reduction_rate = (baseline_storage - optimized_storage) / baseline_storage * 100
        else:
            reduction_rate = 0.0
        
        # 预期降低率
        expected_reduction = {
            "stress_test": 90.0,
            "real_scenario": 45.0,  # 40-50% 的平均值
            "production_simulation": 45.0,
        }.get(scenario_name, 45.0)
        
        scenario_data = {
            "baseline": baseline,
            "optimized": optimized,
            "reduction_rate": reduction_rate,
            "expected_reduction": expected_reduction,
            "meets_expectation": reduction_rate >= expected_reduction * 0.9,  # 允许10%误差
            "knowledge_retention": optimized["knowledge_retention"],
        }
        
        report_data["scenarios"][scenario_name] = scenario_data
        
        # 打印场景报告
        print(f"\n【{scenario_name}】")
        print(f"  输入对话数: {baseline['input_count']}")
        print(f"  对照组存储: {baseline_storage} 条")
        print(f"  实验组存储: {optimized_storage} 条")
        print(f"  存储降低率: {reduction_rate:.2f}% (预期: ≥{expected_reduction}%)")
        print(f"  目标达成: {'✅' if scenario_data['meets_expectation'] else '❌'}")
        print(f"  知识保留度: {optimized['knowledge_retention']['retention_rate']:.2%}")
        if optimized['knowledge_retention']['order_number_found']:
            print(f"    ✓ 订单号信息已保留")
        if optimized['knowledge_retention']['return_intent_found']:
            print(f"    ✓ 退货意图信息已保留")
    
    # 计算平均降低率
    reduction_rates = [
        scenario_data["reduction_rate"]
        for scenario_data in report_data["scenarios"].values()
    ]
    avg_reduction_rate = sum(reduction_rates) / len(reduction_rates) if reduction_rates else 0.0
    
    report_data["summary"] = {
        "average_reduction_rate": avg_reduction_rate,
        "target_achievement": avg_reduction_rate >= 45.0,
    }
    
    print(f"\n【总结】")
    print(f"  平均存储降低率: {avg_reduction_rate:.2f}%")
    print(f"  目标达成: {'✅' if report_data['summary']['target_achievement'] else '❌'} (目标: ≥45%)")
    
    # 保存报告
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
