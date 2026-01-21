"""
长对话测试主脚本

运行完整的长对话性能测试，对比基线系统和增强系统的性能指标。
生成详细的性能报告，验证核心量化目标：
- 长对话成功率提升 30%+（核心指标）
- 存储和延迟：说明功能增强的必要成本
- 未来优化目标：在功能完整的基础上进一步优化
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from scripts.benchmark.baseline_test import BaselineTestRunner

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# 延迟导入增强测试，避免在依赖缺失时无法运行基线测试
try:
    from scripts.benchmark.enhanced_test import EnhancedTestRunner
    ENHANCED_TEST_AVAILABLE = True
except ImportError as e:
    ENHANCED_TEST_AVAILABLE = False
    EnhancedTestRunner = None
    logger.warning(f"增强测试模块不可用: {e}")


class ComparisonReport:
    """对比报告生成器"""
    
    def __init__(self, baseline_results: Dict[str, Any], enhanced_results: Dict[str, Any]):
        """
        初始化报告生成器
        
        Args:
            baseline_results: 基线测试结果
            enhanced_results: 增强系统测试结果
        """
        self.baseline_results = baseline_results
        self.enhanced_results = enhanced_results
    
    def generate_report(self) -> Dict[str, Any]:
        """生成对比报告"""
        report = {
            "timestamp": time.time(),
            "comparisons": {},
            "summary": {},
        }
        
        # 对每个数据集进行对比
        for suite_name in self.baseline_results.keys():
            if suite_name not in self.enhanced_results:
                continue
            
            comparison = self._compare_suite(
                suite_name,
                self.baseline_results[suite_name],
                self.enhanced_results[suite_name],
            )
            report["comparisons"][suite_name] = comparison
        
        # 计算总体摘要
        report["summary"] = self._calculate_summary(report["comparisons"])
        
        return report
    
    def _compare_suite(
        self,
        suite_name: str,
        baseline: Dict[str, Any],
        enhanced: Dict[str, Any],
    ) -> Dict[str, Any]:
        """对比单个数据集的结果"""
        baseline_metrics = baseline["metrics"]
        enhanced_metrics = enhanced["metrics"]
        
        # 计算成功率改进
        # 当基线为0时，计算绝对改进（从0%提升到X%）
        baseline_success = baseline_metrics["success_rate"]
        enhanced_success = enhanced_metrics["success_rate"]
        
        if baseline_success == 0:
            # 基线为0时，计算绝对提升（单位：百分点）
            # 例如：从0%提升到66.67% = +66.67百分点
            success_rate_improvement = enhanced_success * 100
        elif baseline_success > 0:
            # 基线不为0时，计算百分比改进
            success_rate_improvement = (
                (enhanced_success - baseline_success) / baseline_success * 100
            )
        else:
            success_rate_improvement = 0
        
        # 计算存储改进
        # 存储降低是正数，增加是负数
        baseline_storage = baseline_metrics["final_storage_bytes"]
        enhanced_storage = enhanced_metrics["final_storage_bytes"]
        
        if baseline_storage > 0:
            storage_reduction = (
                (baseline_storage - enhanced_storage) / baseline_storage * 100
            )
        else:
            storage_reduction = 0
        
        # 计算延迟缩短
        # 延迟缩短是正数，增加是负数
        baseline_latency = (
            baseline_metrics["avg_retrieval_latency"] + 
            baseline_metrics["avg_generation_latency"]
        )
        enhanced_latency = (
            enhanced_metrics["avg_retrieval_latency"] + 
            enhanced_metrics["avg_generation_latency"]
        )
        
        if baseline_latency > 0:
            latency_reduction = (
                (baseline_latency - enhanced_latency) / baseline_latency * 100
            )
        else:
            latency_reduction = 0
        
        return {
            "baseline": baseline_metrics,
            "enhanced": enhanced_metrics,
            "improvements": {
                "success_rate_improvement": success_rate_improvement,
                "storage_reduction": storage_reduction,
                "latency_reduction": latency_reduction,
            },
            "targets_met": {
                "success_rate_target_30pct": success_rate_improvement >= 30,
                "storage_target_45pct": storage_reduction >= 45,
                "latency_target_25pct": latency_reduction >= 25,
            },
        }
    
    def _calculate_summary(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """计算总体摘要"""
        if not comparisons:
            return {}
        
        # 计算平均改进
        improvements = [c["improvements"] for c in comparisons.values()]
        
        avg_success_improvement = sum(i["success_rate_improvement"] for i in improvements) / len(improvements)
        avg_storage_reduction = sum(i["storage_reduction"] for i in improvements) / len(improvements)
        avg_latency_reduction = sum(i["latency_reduction"] for i in improvements) / len(improvements)
        
        # 计算目标达成情况
        targets_met = [c["targets_met"] for c in comparisons.values()]
        success_target_met = sum(1 for t in targets_met if t["success_rate_target_30pct"]) / len(targets_met) * 100
        storage_target_met = sum(1 for t in targets_met if t["storage_target_45pct"]) / len(targets_met) * 100
        latency_target_met = sum(1 for t in targets_met if t["latency_target_25pct"]) / len(targets_met) * 100
        
        return {
            "average_improvements": {
                "success_rate_improvement": avg_success_improvement,
                "storage_reduction": avg_storage_reduction,
                "latency_reduction": avg_latency_reduction,
            },
            "target_achievement_rate": {
                "success_rate_30pct": success_target_met,
                "storage_45pct": storage_target_met,
                "latency_25pct": latency_target_met,
            },
        }
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """打印报告"""
        print("\n" + "=" * 80)
        print("性能对比报告")
        print("=" * 80)
        
        # 打印详细对比
        for suite_name, comparison in report["comparisons"].items():
            print(f"\n【{suite_name}】")
            print("-" * 80)
            
            improvements = comparison["improvements"]
            targets = comparison["targets_met"]
            baseline = comparison["baseline"]
            enhanced = comparison["enhanced"]
            
            baseline_success = baseline["success_rate"]
            enhanced_success = enhanced["success_rate"]
            
            # 格式化成功率改进显示
            if baseline_success == 0:
                # 基线为0时，显示绝对提升
                success_display = f"{improvements['success_rate_improvement']:+.2f} 百分点"
                success_note = f"（从 {baseline_success*100:.1f}% 提升到 {enhanced_success*100:.1f}%）"
            else:
                # 基线不为0时，显示百分比改进
                success_display = f"{improvements['success_rate_improvement']:+.2f}%"
                success_note = f"（从 {baseline_success*100:.1f}% 提升到 {enhanced_success*100:.1f}%）"
            
            print(f"  成功率改进: {success_display} {success_note} "
                  f"{'✅' if targets['success_rate_target_30pct'] else '❌'}")
            print(f"  存储成本降低: {improvements['storage_reduction']:+.2f}% "
                  f"{'✅' if targets['storage_target_45pct'] else '❌'}")
            print(f"  延迟缩短: {improvements['latency_reduction']:+.2f}% "
                  f"{'✅' if targets['latency_target_25pct'] else '❌'}")
            
            print(f"\n  基线系统指标:")
            print(f"    成功率: {baseline['success_rate']:.2%}")
            print(f"    存储: {baseline['final_storage_bytes'] / 1024:.2f} KB")
            print(f"    延迟: {baseline['avg_retrieval_latency'] + baseline['avg_generation_latency']:.3f}s")
            
            print(f"\n  增强系统指标:")
            print(f"    成功率: {enhanced['success_rate']:.2%}")
            print(f"    存储: {enhanced['final_storage_bytes'] / 1024:.2f} KB")
            print(f"    延迟: {enhanced['avg_retrieval_latency'] + enhanced['avg_generation_latency']:.3f}s")
        
        # 打印摘要
        print("\n" + "=" * 80)
        print("总体摘要")
        print("=" * 80)
        
        summary = report["summary"]
        if summary:
            avg_imp = summary["average_improvements"]
            target_ach = summary["target_achievement_rate"]
            
            print(f"\n平均改进:")
            print(f"  成功率提升: {avg_imp['success_rate_improvement']:+.2f}% "
                  f"(目标: ≥30%, 达成率: {target_ach['success_rate_30pct']:.1f}%)")
            print(f"  存储成本降低: {avg_imp['storage_reduction']:+.2f}% "
                  f"(目标: ≥45%, 达成率: {target_ach['storage_45pct']:.1f}%)")
            print(f"  延迟缩短: {avg_imp['latency_reduction']:+.2f}% "
                  f"(目标: ≥25%, 达成率: {target_ach['latency_25pct']:.1f}%)")
        
        print()


def main():
    """主函数"""
    print("=== 长对话性能测试套件 ===\n")
    
    output_dir = Path(__file__).parent.parent / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 运行基线测试
    print("步骤 1/3: 运行基线系统测试...")
    baseline_runner = BaselineTestRunner(output_dir)
    
    # 检查是否已有基线结果
    baseline_file = output_dir / "baseline_results.json"
    if baseline_file.exists():
        print(f"  发现已有基线测试结果: {baseline_file}")
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_results_all = json.load(f)
        # 使用滑动窗口作为主要对比基线
        baseline_results = baseline_results_all.get("sliding_window", {})
        if not baseline_results:
            print("  未找到滑动窗口结果，重新运行测试...")
            baseline_results_all = baseline_runner.run_baseline_suite()
            baseline_results = baseline_results_all["sliding_window"]
    else:
        print("  运行基线测试...")
        baseline_results_all = baseline_runner.run_baseline_suite()
        baseline_results = baseline_results_all["sliding_window"]
    
    print("  基线测试完成\n")
    
    # 2. 运行增强系统测试
    if not ENHANCED_TEST_AVAILABLE:
        print("步骤 2/3: 增强系统测试不可用（缺少依赖）")
        print("  提示: 要运行增强系统测试，请先安装依赖:")
        print("    pip install langchain-core langchain-community")
        print("  现在仅显示基线测试结果\n")
        
        print("\n=== 基线测试完成 ===")
        print(f"  基线结果: {baseline_file}")
        print("  提示: 安装依赖后运行增强测试: python scripts/benchmark/enhanced_test.py")
        return
    
    print("步骤 2/3: 运行增强系统测试...")
    enhanced_runner = EnhancedTestRunner(output_dir)
    
    # 检查是否已有增强结果
    enhanced_file = output_dir / "enhanced_results.json"
    if enhanced_file.exists():
        print(f"  发现已有增强系统测试结果: {enhanced_file}")
        with open(enhanced_file, "r", encoding="utf-8") as f:
            enhanced_results = json.load(f)
    else:
        print("  运行增强系统测试...")
        enhanced_results = enhanced_runner.run_enhanced_suite()
    
    print("  增强系统测试完成\n")
    
    # 3. 生成对比报告
    print("步骤 3/3: 生成对比报告...")
    report_generator = ComparisonReport(baseline_results, enhanced_results)
    report = report_generator.generate_report()
    
    # 保存报告
    report_file = output_dir / "comparison_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印报告
    report_generator.print_report(report)
    
    print("\n=== 测试完成 ===")
    print(f"  基线结果: {baseline_file}")
    print(f"  增强结果: {enhanced_file}")
    print(f"  对比报告: {report_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)
