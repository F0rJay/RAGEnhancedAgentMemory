"""
Baseline Auto-Player: 基线对照组脚本
用于对比测试：不使用 RAGEnhancedAgentMemory 插件，其他条件相同

对比维度：
1. 推理性能（延迟、TTFT）
2. 存储效率（无优化，所有对话都存储）
3. 长期记忆召回能力（无长期记忆，无法召回早期信息）
"""

import sys
import time
import uuid
import statistics
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque
from rich.console import Console
from rich.table import Table
from rich.progress import track

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 在导入任何库之前设置 HuggingFace 镜像源（如果配置了）
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        try:
            from huggingface_hub import set_endpoint
            set_endpoint(hf_endpoint)
        except ImportError:
            pass
except Exception:
    pass

# 导入本地推理引擎（基线版使用 BaselineInference）
try:
    from src.inference import BaselineInference, TRANSFORMERS_AVAILABLE
    from src.config import get_settings
    LOCAL_INFERENCE_AVAILABLE = TRANSFORMERS_AVAILABLE
except ImportError as e:
    LOCAL_INFERENCE_AVAILABLE = False
    BaselineInference = None
    print(f"警告: 无法导入基线推理引擎: {e}")
    print("请确保已安装 transformers: pip install transformers torch")

# 初始化 Rich 控制台
console = Console()


class BaselineAgent:
    """
    基线 Agent（不使用 RAGEnhancedAgentMemory）
    
    使用简单的滑动窗口记忆，只保留最近 N 轮对话
    不进行语义检索、去重、过滤等优化
    """
    
    def __init__(self, window_size: int = 10):
        """
        初始化基线 Agent
        
        Args:
            window_size: 滑动窗口大小（保留的对话轮数）
        """
        self.window_size = window_size
        self.conversation_history = deque(maxlen=window_size)
        self.total_turns = 0
        self.session_id = f"baseline_{uuid.uuid4().hex[:8]}"
        
    def add_conversation(self, human: str, ai: str) -> None:
        """添加对话到历史（滑动窗口）"""
        self.total_turns += 1
        self.conversation_history.append({
            "human": human,
            "ai": ai,
            "turn": self.total_turns
        })
    
    def get_context(self, query: str) -> List[str]:
        """
        获取当前上下文（只包含窗口内的对话）
        
        Args:
            query: 当前查询
            
        Returns:
            上下文列表
        """
        context = []
        for turn in self.conversation_history:
            context.append(f"用户: {turn['human']}\n助手: {turn['ai']}")
        return context
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_turns": self.total_turns,
            "window_size": self.window_size,
            "current_history_size": len(self.conversation_history),
            "has_long_term_memory": False,
        }


class BaselineAutoPlayer:
    def __init__(self, model_path: Optional[str] = None, window_size: int = 10):
        """
        初始化基线 Auto-Player
        
        基线系统使用简单的滑动窗口记忆（不使用 RAGEnhancedAgentMemory），
        唯一的区别是：
        - 增强版：使用 RAGEnhancedAgentMemory + VLLMInference
        - 基线版：使用滑动窗口记忆 + BaselineInference
        
        但两者使用相同的模型（都是用户选择的）
        
        Args:
            model_path: 模型路径或名称（如果为 None，从配置读取）
            window_size: 滑动窗口大小（保留的对话轮数）
        """
        self.settings = get_settings()
        self.session_id = f"baseline_{uuid.uuid4().hex[:8]}"
        self.agent = BaselineAgent(window_size=window_size)
        
        # 初始化本地推理引擎（基线版使用 BaselineInference）
        # 基线系统强制使用本地 transformers 模型（HuggingFace 模型或本地路径），不使用 API
        # 优先使用传入的 model_path，如果没有则使用配置中的路径
        if model_path:
            self.model_path = model_path
        else:
            # 优先使用新的配置项，其次使用已废弃的配置项
            self.model_path = self.settings.vllm_model or self.settings.vllm_model_path
        
        # 基线系统必须使用本地模型，不能使用 API
        # 如果仍然没有模型路径，报错提示用户配置
        if not self.model_path:
            console.print("[red]❌ 错误: 模型路径未设置[/red]")
            console.print("[yellow]提示: 请通过以下方式之一指定模型：[/yellow]")
            console.print("[dim]  1. 命令行参数: --model-path <model-name> 或 <local-path>[/dim]")
            console.print("[dim]  2. 环境变量: 在 .env 文件中设置 VLLM_MODEL=<model-name>[/dim]")
            console.print("[dim]  3. 基线系统必须使用本地模型，不支持 API[/dim]")
            raise ValueError("模型路径未设置，无法初始化基线推理引擎")
        
        # 初始化基线推理引擎（BaselineInference，强制使用本地 transformers 模式）
        if LOCAL_INFERENCE_AVAILABLE or True:
            try:
                console.print(f"[yellow]📊 使用基线推理引擎（BaselineInference，本地 transformers 模式）[/yellow]")
                console.print(f"[dim]模型: {self.model_path}[/dim]")
                
                # 基线系统强制使用本地 transformers 模型，不使用 API
                # BaselineInference 会自动检测：如果 base_url 是本地 vLLM 服务且模型路径是本地路径，
                # 则强制使用本地 transformers 模式（不通过 vLLM API）
                self.inference_engine = BaselineInference(
                    model_path=self.model_path,
                    base_url=self.settings.vllm_base_url,  # 传递 base_url，但 BaselineInference 会判断使用本地模式
                    api_key=self.settings.vllm_api_key,
                    timeout=self.settings.vllm_timeout
                )
                self.use_llm = True
                
                # 显示使用的模式
                if hasattr(self.inference_engine, 'use_api') and self.inference_engine.use_api:
                    console.print(f"[dim]模式: API（{self.inference_engine.base_url}）[/dim]")
                else:
                    console.print(f"[dim]模式: 本地模型[/dim]")
            except Exception as e:
                console.print(f"[red]❌ 基线推理引擎初始化失败: {e}[/red]")
                import traceback
                traceback.print_exc()
                self.inference_engine = None
                self.use_llm = False
                console.print("\n[red]❌ 基线系统必须使用 BaselineInference，程序退出[/red]")
                sys.exit(1)
        else:
            console.print("[red]❌ 无法使用基线推理引擎[/red]")
            console.print("[dim]提示: 请安装必要的依赖（transformers 或 openai）[/dim]")
            self.inference_engine = None
            self.use_llm = False
            console.print("\n[red]❌ 基线系统必须使用 BaselineInference，程序退出[/red]")
            sys.exit(1)
        
        # 性能记录
        self.latencies = []
        self.ttfts = []  # 首字延迟
        self.tokens_per_second = []  # 每秒 tokens（吞吐量）
        self.tokens_generated = []  # 每次生成的 tokens 数
        self.tokens_per_second = []  # 每秒 tokens（吞吐量）
        self.tokens_generated = []  # 每次生成的 tokens 数
        
        # 长期记忆召回测试记录（基线系统无法召回早期信息）
        self.recall_test_cases = {
            "工号": {"query": "我的工号", "expected": "9527", "found": False, "retrieved_content": ""},
            "咖啡习惯": {"query": "我喝咖啡的习惯", "expected": "冰美式，不加糖", "found": False, "retrieved_content": ""},
            "订单号": {"query": "查询的订单号", "expected": "ORDER_2026_001", "found": False, "retrieved_content": ""},
        }
    
    def generate_script(self) -> List[str]:
        """生成 100 轮模拟剧本（与增强版相同）"""
        script = []
        
        # Phase 1: 设定人设 (1-5轮)
        script.extend([
            "你好，我是测试员阿强。我的工号是 9527。",
            "我喜欢喝冰美式，不加糖。",
            "我们今天的任务是测试 Agent 的极限性能。",
            "你只需要简短回复收到即可。",
            "准备好了吗？"
        ])
        
        # Phase 2: 疯狂复读 (6-35轮)
        for _ in range(10):
            script.append("帮我查一下订单 ORDER_2026_001 的状态")
            script.append("订单 ORDER_2026_001 发货了吗？")
            script.append("还没查到 ORDER_2026_001 吗？")
            
        # Phase 3: 低价值灌水 (36-85轮)
        fillers = ["好的", "收到", "明白", "OK", "谢谢", "在吗", "嗯嗯", "不错", "哈哈", "确实"]
        for i in range(50):
            script.append(fillers[i % len(fillers)])
            
        # Phase 4: 记忆突击检查 (86-100轮)
        script.extend([
            "测试快结束了，稍微总结一下。",
            "对了，还记得我是谁吗？报上我的工号。",
            "我刚才说我喝咖啡有什么习惯？",
            "我们刚才查询了哪个订单号？",
            "这一百句聊下来，你累不累？",
            "再见！"
        ])
        
        return script
    
    def call_llm(self, user_input: str, context: List[str]) -> tuple[str, float]:
        """
        调用基线推理引擎生成回复（使用 BaselineInference）
        
        Args:
            user_input: 用户输入
            context: 上下文列表（基线系统只有滑动窗口内的对话）
            
        Returns:
            (生成的回复, 总延迟时间(ms))
        """
        if not self.use_llm or not self.inference_engine:
            console.print("[red]❌ 推理引擎不可用，程序退出[/red]")
            sys.exit(1)
        
        # 构建提示词（与增强版相同）
        system_prompt = "你是一个专业的AI助手，能够根据上下文信息准确回答用户问题。请简洁、准确地回复。"
        
        # 构建上下文（基线系统只有滑动窗口内的对话）
        context_text = ""
        if context:
            context_text = "\n".join([f"- {doc}" for doc in context[:5]])  # 最多使用5条上下文
            context_text = f"相关上下文：\n{context_text}\n\n"
        
        user_prompt = f"{context_text}用户问题：{user_input}"
        
        try:
            # 调用基线推理引擎
            response, metrics = self.inference_engine.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=512  # max_model_len=1024，可以设置更大的值
            )
            
            # 提取延迟信息（与增强版相同的指标）
            total_time = metrics.get("total_time", 0) * 1000  # 转换为 ms
            ttft = metrics.get("ttft", 0) * 1000  # 转换为 ms
            tps = metrics.get("tokens_per_second", 0)  # 每秒 tokens
            tokens = metrics.get("tokens_generated", 0)  # 生成的 tokens 数
            
            # 记录性能指标（与增强版相同）
            if ttft > 0:
                self.ttfts.append(ttft)
            if tps > 0:
                self.tokens_per_second.append(tps)
            if tokens > 0:
                self.tokens_generated.append(tokens)
            
            return response, total_time
            
        except Exception as e:
            console.print(f"[red]❌ 基线推理引擎调用失败: {e}[/red]")
            import traceback
            traceback.print_exc()
            console.print("\n[red]❌ 推理引擎调用失败，程序退出[/red]")
            sys.exit(1)
    
    def _test_recall(self, user_input: str, context_docs: List[str], response: str):
        """
        测试召回能力（基线系统只有滑动窗口内的对话，无法召回早期信息）
        
        Args:
            user_input: 用户输入
            context_docs: 上下文文档（基线系统只有窗口内的对话）
            response: Agent 回复
        """
        # 合并所有文本用于检查
        all_text = " ".join(context_docs) + " " + response
        
        # 检查工号召回（第1轮的信息，窗口大小为10时已不在窗口内）
        if ("工号" in user_input or "我是谁" in user_input) and not self.recall_test_cases["工号"]["found"]:
            if "9527" in all_text:
                self.recall_test_cases["工号"]["found"] = True
                for doc in context_docs:
                    if "9527" in doc:
                        self.recall_test_cases["工号"]["retrieved_content"] = doc[:150]
                        break
                if not self.recall_test_cases["工号"]["retrieved_content"]:
                    self.recall_test_cases["工号"]["retrieved_content"] = response[:150]
        
        # 检查咖啡习惯召回（第2轮的信息）
        if ("咖啡" in user_input or ("喝" in user_input and "习惯" in user_input)) and not self.recall_test_cases["咖啡习惯"]["found"]:
            if "冰美式" in all_text or "不加糖" in all_text:
                self.recall_test_cases["咖啡习惯"]["found"] = True
                for doc in context_docs:
                    if "冰美式" in doc or "不加糖" in doc:
                        self.recall_test_cases["咖啡习惯"]["retrieved_content"] = doc[:150]
                        break
                if not self.recall_test_cases["咖啡习惯"]["retrieved_content"]:
                    self.recall_test_cases["咖啡习惯"]["retrieved_content"] = response[:150]
        
        # 检查订单号召回（第6-35轮的信息，可能在窗口内）
        if ("订单" in user_input and "哪个" in user_input) and not self.recall_test_cases["订单号"]["found"]:
            if "ORDER_2026_001" in all_text or "ORDER_2024_001" in all_text:
                self.recall_test_cases["订单号"]["found"] = True
                for doc in context_docs:
                    if "ORDER_2026_001" in doc or "ORDER_2024_001" in doc:
                        self.recall_test_cases["订单号"]["retrieved_content"] = doc[:150]
                        break
                if not self.recall_test_cases["订单号"]["retrieved_content"]:
                    self.recall_test_cases["订单号"]["retrieved_content"] = response[:150]

    def run(self):
        script = self.generate_script()
        console.print(f"[bold yellow]📊 Baseline Auto-Player 启动（对照组）！目标: {len(script)} 轮对话[/bold yellow]")
        console.print(f"Session ID: {self.session_id}")
        
        if not self.use_llm:
            console.print("[red]❌ 推理引擎未初始化，程序退出[/red]")
            sys.exit(1)
        
        llm_status = "✅ BaselineInference" if self.use_llm else "❌ 未初始化"
        console.print(f"配置: 滑动窗口大小={self.agent.window_size}, LLM={llm_status}, 模型={self.model_path}")
        console.print(f"[dim]⚠️  基线系统：无长期记忆、无语义检索、无存储优化、无 vLLM 优化[/dim]")
        print("-" * 60)

        # 基线系统没有向量数据库，记录对话历史大小
        start_history_size = len(self.agent.conversation_history)

        for i, user_input in enumerate(track(script, description="正在交互（基线）...")):
            start_time = time.time()
            
            # 获取上下文（只有滑动窗口内的对话）
            context_docs = self.agent.get_context(user_input)
            
            # 调用 LLM 生成回复
            response, llm_latency = self.call_llm(user_input, context_docs)
            
            # 保存对话到滑动窗口
            self.agent.add_conversation(user_input, response)
            
            end_time = time.time()
            total_latency = (end_time - start_time) * 1000  # ms
            
            self.latencies.append(total_latency)
            
            # 测试召回能力（Phase 4）
            if i >= 85:
                self._test_recall(user_input, context_docs, response)
            
            # 简单打印交互
            if i < 5 or i > 85 or i % 20 == 0:
                console.print(f"[blue]User({i+1}):[/blue] {user_input}")
                ttft_info = f", TTFT: {self.ttfts[-1]:.1f}ms" if self.ttfts else ""
                console.print(f"[yellow]Agent:[/yellow] {response[:80]}... [dim](总延迟: {total_latency:.1f}ms{ttft_info}, 窗口内对话{len(context_docs)}条)[/dim]")

        # 获取最终状态
        end_history_size = len(self.agent.conversation_history)
        stats = self.agent.get_stats()
        total_turns = len(script)
        stored_turns = end_history_size
        
        # 计算存储统计
        # 基线系统：所有对话都保存在窗口内（超出窗口的丢失）
        # 噪音过滤率：基线系统没有过滤，所以过滤率为0
        noise_filter_rate = 0.0
        
        # 信息保留率：只保留窗口内的对话
        retention_rate = (stored_turns / total_turns * 100) if total_turns > 0 else 0
        
        self.generate_report(len(script), start_history_size, end_history_size, stats, 
                           stored_turns, noise_filter_rate, retention_rate)

    def generate_report(self, total_turns, start_history_size, end_history_size, stats,
                       stored_turns, noise_filter_rate, retention_rate):
        """
        生成报告
        
        Args:
            total_turns: 总对话轮数
            start_history_size: 初始历史大小
            end_history_size: 最终历史大小
            stats: 统计信息
            stored_turns: 存储的对话数
            noise_filter_rate: 噪音过滤率
            retention_rate: 信息保留率
        """
        console.print("\n\n")
        console.rule("[bold yellow]📊 基线系统验证报告（对照组）[/bold yellow]")
        
        # 1. 性能指标（与增强版相同）
        avg_latency = statistics.mean(self.latencies) if self.latencies else 0
        p95_latency = statistics.quantiles(self.latencies, n=20)[18] if len(self.latencies) >= 20 else 0
        
        perf_table = Table(title="⚡ 推理性能 (Inference Speed)")
        perf_table.add_column("指标", style="cyan")
        perf_table.add_column("结果", style="green")
        perf_table.add_column("评价")
        
        # 计算首字延迟统计
        avg_ttft = statistics.mean(self.ttfts) if self.ttfts else 0
        p95_ttft = statistics.quantiles(self.ttfts, n=20)[18] if len(self.ttfts) >= 20 else 0
        
        perf_table.add_row("平均延迟", f"{avg_latency:.1f} ms", "🚀 秒回" if avg_latency < 500 else "⚠️ 正常")
        perf_table.add_row("P95 延迟", f"{p95_latency:.1f} ms", "在高负载下表现稳定")
        if avg_ttft > 0:
            perf_table.add_row("平均首字延迟 (TTFT)", f"{avg_ttft:.1f} ms", "🚀 极快" if avg_ttft < 200 else "✅ 良好")
            perf_table.add_row("P95 首字延迟", f"{p95_ttft:.1f} ms", "高负载下首字响应稳定")
        
        # 吞吐量指标
        avg_tps = statistics.mean(self.tokens_per_second) if self.tokens_per_second else 0
        p95_tps = statistics.quantiles(self.tokens_per_second, n=20)[18] if len(self.tokens_per_second) >= 20 else 0
        
        if avg_tps > 0:
            perf_table.add_row("平均吞吐量", f"{avg_tps:.1f} tokens/s", "🚀 优秀" if avg_tps > 50 else "✅ 良好")
            perf_table.add_row("P95 吞吐量", f"{p95_tps:.1f} tokens/s", "高负载下吞吐量稳定")
        
        perf_table.add_row("总耗时", f"{sum(self.latencies)/1000:.1f} s", f"处理 {total_turns} 轮对话")
        
        console.print(perf_table)
        console.print("\n")

        # 2. 存储指标（基线系统：所有对话都保存在窗口内，超出窗口的丢失）
        lost_turns = total_turns - stored_turns
        
        store_table = Table(title="💾 存储情况 (Storage)")
        store_table.add_column("指标", style="cyan")
        store_table.add_column("数据", style="magenta")
        store_table.add_column("说明")
        
        store_table.add_row("输入对话总数", str(total_turns), "模拟的用户输入")
        store_table.add_row("窗口中存储数", str(stored_turns), f"窗口内保留的对话数（窗口大小: {stats['window_size']}）")
        store_table.add_row("丢失的对话数", str(lost_turns), f"超出窗口的对话已丢失")
        store_table.add_row("噪音过滤率", f"{noise_filter_rate:.1f}%", "基线系统无过滤功能，所有对话都存储（在窗口内）")
        store_table.add_row("信息保留率", f"{retention_rate:.1f}%", "仅保留最近窗口内的对话")
        
        console.print(store_table)
        console.print("\n")
        
        # 3. 长期记忆召回能力（基线系统无法召回早期信息）
        recall_table = Table(title="🧠 长期记忆召回能力 (Long-Term Memory Recall)")
        recall_table.add_column("测试项", style="cyan", width=12)
        recall_table.add_column("期望信息", style="yellow", width=20)
        recall_table.add_column("召回状态", style="green", width=12)
        recall_table.add_column("检索内容预览", style="dim", width=40)
        
        recall_success_count = 0
        for test_name, test_case in self.recall_test_cases.items():
            status = "✅ 成功" if test_case["found"] else "❌ 失败"
            status_style = "green" if test_case["found"] else "red"
            if test_case["found"]:
                recall_success_count += 1
            
            retrieved_preview = test_case["retrieved_content"][:50] + "..." if test_case["retrieved_content"] else "[dim]未检索到（超出窗口）[/dim]"
            
            recall_table.add_row(
                test_name,
                test_case["expected"],
                f"[{status_style}]{status}[/{status_style}]",
                retrieved_preview
            )
        
        recall_rate = (recall_success_count / len(self.recall_test_cases)) * 100 if self.recall_test_cases else 0
        
        console.print(recall_table)
        
        console.print(f"\n[bold cyan]📊 召回能力分析：[/bold cyan]")
        console.print(f"  • 测试场景：在 {total_turns} 轮对话后，测试系统是否能召回早期（第1-5轮）的关键信息")
        console.print(f"  • 召回成功率：{recall_rate:.1f}% ({recall_success_count}/{len(self.recall_test_cases)})")
        console.print(f"  • 基线系统限制：滑动窗口大小 {stats['window_size']}，超出窗口的对话已丢失")
        console.print(f"  • 预期结果：早期信息（工号、咖啡习惯）无法召回，因为已超出窗口")
        console.print()
        
        # 4. 结论
        console.rule("[bold]📝 最终结论[/bold]")
        if avg_latency < 2000:
            console.print("✅ [bold green]基线推理正常：[/bold green] 响应速度良好。")
        else:
            console.print("⚠️ [bold yellow]基线延迟较高：[/bold yellow] 请检查 GPU 配置或模型路径。")
        
        if avg_ttft > 0 and avg_ttft < 500:
            console.print(f"✅ [bold green]首字延迟良好：[/bold green] 平均 TTFT {avg_ttft:.1f}ms。")
        elif avg_ttft > 0:
            console.print(f"⚠️ [bold yellow]首字延迟：[/bold yellow] 平均 TTFT {avg_ttft:.1f}ms，可能需要优化配置。")
        
        console.print("⚠️ [bold yellow]基线系统限制：[/bold yellow] 无长期记忆、无语义检索、无存储优化，所有对话都存储在滑动窗口内。")
        
        # 长期记忆召回能力评价
        if recall_rate >= 100:
            console.print(f"✅ [bold green]召回能力良好：[/bold green] 召回成功率 {recall_rate:.1f}%，所有关键信息都能正确召回。")
        elif recall_rate > 0:
            console.print(f"⚠️ [bold yellow]召回能力有限：[/bold yellow] 召回成功率 {recall_rate:.1f}%，只能召回窗口内的信息。")
        else:
            console.print(f"❌ [bold red]召回能力不足：[/bold red] 召回成功率 0%，早期信息已超出窗口，无法召回。")

    def close(self) -> None:
        """清理资源"""
        pass  # 基线系统无需清理


if __name__ == "__main__":
    import argparse
    import atexit
    import signal
    
    def cleanup_resources():
        """清理资源"""
        try:
            if player is not None and hasattr(player, 'close'):
                player.close()
        except Exception:
            pass
    
    parser = argparse.ArgumentParser(description="Baseline Auto-Player: 基线对照组脚本")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="本地模型路径（也可通过环境变量 VLLM_MODEL_PATH 设置，默认: Qwen/Qwen2.5-7B-Instruct）"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="滑动窗口大小（默认: 10）"
    )
    
    args = parser.parse_args()
    
    player = None
    try:
        player = BaselineAutoPlayer(
            model_path=args.model_path,
            window_size=args.window_size
        )
        
        atexit.register(cleanup_resources)
        
        def signal_handler(sig, frame):
            cleanup_resources()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        player.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  用户中断[/yellow]")
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]❌ 运行出错: {e}[/red]")
        cleanup_resources()
        sys.exit(1)
    finally:
        cleanup_resources()
