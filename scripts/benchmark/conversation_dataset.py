"""
长对话测试数据集生成器

生成用于测试的对话数据集，包含：
- 早期关键信息（测试记忆保持能力）
- 中期话题切换
- 后期对早期信息的引用（测试检索能力）
"""

import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ConversationExample:
    """对话示例"""
    turn_number: int
    human_message: str
    ai_message: str
    contains_key_info: bool = False  # 是否包含关键信息
    key_info_type: Optional[str] = None  # 关键信息类型
    references_early_info: bool = False  # 是否引用了早期信息
    referenced_turn: Optional[int] = None  # 引用的轮次


@dataclass
class TestDataset:
    """测试数据集"""
    name: str
    conversations: List[ConversationExample]
    key_info: Dict[int, str]  # 轮次 -> 关键信息
    test_questions: List[Dict[str, Any]]  # 测试问题（用于验证记忆）
    total_turns: int


class ConversationDatasetGenerator:
    """对话数据集生成器"""
    
    # 预定义的话题模板
    TOPICS = {
        "个人信息": [
            ("我的名字是{name}，来自{city}。", "很高兴认识你，{name}！{city}是个好地方。"),
            ("我今年{age}岁，是一名{job}。", "原来如此，{job}是很有意义的职业。"),
            ("我最喜欢的颜色是{color}。", "{color}是个很棒的颜色选择。"),
        ],
        "技术偏好": [
            ("我喜欢用{language}编程。", "{language}是个优秀的编程语言。"),
            ("我使用{framework}框架开发。", "{framework}确实很流行。"),
            ("我的IDE是{tool}。", "{tool}是个强大的开发工具。"),
        ],
        "日常对话": [
            ("今天天气真好。", "是的，很适合出门走走。"),
            ("你最近怎么样？", "我很好，谢谢关心。"),
            ("有什么推荐的吗？", "根据你的兴趣，我建议..."),
        ],
        "深度讨论": [
            ("关于{topic}，你怎么看？", "这个话题很有意思，我认为..."),
            ("能详细解释一下{concept}吗？", "当然，{concept}是指..."),
        ],
    }
    
    def __init__(self, seed: Optional[int] = None):
        """初始化生成器"""
        if seed is not None:
            random.seed(seed)
        logger.info(f"初始化对话数据集生成器 (seed={seed})")
    
    def generate_long_conversation(
        self,
        total_turns: int,
        key_info_turns: List[int],
        test_scenarios: Optional[List[str]] = None,
    ) -> TestDataset:
        """
        生成长对话数据集
        
        Args:
            total_turns: 总对话轮数
            key_info_turns: 包含关键信息的轮次列表（用于后续测试）
            test_scenarios: 测试场景列表
        
        Returns:
            测试数据集
        """
        conversations = []
        key_info_map = {}
        
        # 生成随机用户信息
        user_info = self._generate_user_info()
        
        # 生成对话
        for turn_num in range(1, total_turns + 1):
            # 判断是否包含关键信息
            is_key_turn = turn_num in key_info_turns
            
            if is_key_turn:
                # 生成包含关键信息的对话
                human_msg, ai_msg, key_info = self._generate_key_info_turn(
                    turn_num, user_info
                )
                key_info_map[turn_num] = key_info
                contains_key = True
                info_type = "user_info"
            else:
                # 生成普通对话
                human_msg, ai_msg = self._generate_normal_turn(
                    turn_num, user_info, conversations
                )
                contains_key = False
                info_type = None
                key_info = None
            
            # 检查是否引用了早期信息
            references_early, ref_turn = self._check_early_reference(
                human_msg, turn_num, conversations
            )
            
            conv = ConversationExample(
                turn_number=turn_num,
                human_message=human_msg,
                ai_message=ai_msg,
                contains_key_info=contains_key,
                key_info_type=info_type,
                references_early_info=references_early,
                referenced_turn=ref_turn,
            )
            conversations.append(conv)
        
        # 生成测试问题
        test_questions = self._generate_test_questions(
            conversations, key_info_map, total_turns
        )
        
        dataset = TestDataset(
            name=f"long_conversation_{total_turns}_turns",
            conversations=conversations,
            key_info=key_info_map,
            test_questions=test_questions,
            total_turns=total_turns,
        )
        
        logger.info(
            f"生成长对话数据集: {total_turns} 轮, "
            f"{len(key_info_map)} 个关键信息点, "
            f"{len(test_questions)} 个测试问题"
        )
        
        return dataset
    
    def _generate_user_info(self) -> Dict[str, str]:
        """生成随机用户信息"""
        names = ["张三", "李四", "王五", "赵六", "钱七"]
        cities = ["北京", "上海", "深圳", "杭州", "广州"]
        jobs = ["软件工程师", "数据科学家", "产品经理", "设计师", "研究员"]
        languages = ["Python", "Java", "JavaScript", "Go", "Rust"]
        frameworks = ["Django", "Flask", "React", "Vue", "Spring"]
        colors = ["蓝色", "绿色", "红色", "紫色", "橙色"]
        
        return {
            "name": random.choice(names),
            "city": random.choice(cities),
            "age": str(random.randint(25, 40)),
            "job": random.choice(jobs),
            "language": random.choice(languages),
            "framework": random.choice(frameworks),
            "color": random.choice(colors),
            "tool": "VSCode",
        }
    
    def _generate_key_info_turn(
        self, turn_num: int, user_info: Dict[str, str]
    ) -> Tuple[str, str, str]:
        """生成包含关键信息的对话轮次"""
        # 根据轮次位置选择不同的关键信息
        info_types = [
            ("name", "city"),
            ("age", "job"),
            ("language", "framework"),
            ("color",),
        ]
        
        info_type_group = info_types[(turn_num - 1) % len(info_types)]
        
        # 构建包含关键信息的问题
        if "name" in info_type_group:
            human_msg = f"你好，我的名字是{user_info['name']}，我来自{user_info['city']}。"
            ai_msg = f"很高兴认识你，{user_info['name']}！{user_info['city']}是个很棒的城市。"
            key_info = f"用户姓名: {user_info['name']}, 来自: {user_info['city']}"
        
        elif "age" in info_type_group:
            human_msg = f"我今年{user_info['age']}岁，是一名{user_info['job']}。"
            ai_msg = f"原来如此，{user_info['job']}是很有意义的职业。"
            key_info = f"年龄: {user_info['age']}, 职业: {user_info['job']}"
        
        elif "language" in info_type_group:
            human_msg = f"我喜欢用{user_info['language']}编程，主要使用{user_info['framework']}框架。"
            ai_msg = f"{user_info['language']}配合{user_info['framework']}是个很好的技术栈。"
            key_info = f"编程语言: {user_info['language']}, 框架: {user_info['framework']}"
        
        else:
            human_msg = f"我最喜欢的颜色是{user_info['color']}。"
            ai_msg = f"{user_info['color']}是个很棒的颜色选择。"
            key_info = f"喜欢的颜色: {user_info['color']}"
        
        return human_msg, ai_msg, key_info
    
    def _generate_normal_turn(
        self,
        turn_num: int,
        user_info: Dict[str, str],
        previous_conversations: List[ConversationExample],
    ) -> Tuple[str, str]:
        """生成普通对话轮次"""
        # 随机选择话题
        topic = random.choice(list(self.TOPICS.keys()))
        templates = self.TOPICS[topic]
        template = random.choice(templates)
        
        # 填充模板
        if "{topic}" in template[0]:
            topics = ["人工智能", "机器学习", "深度学习", "自然语言处理"]
            human_msg = template[0].format(topic=random.choice(topics))
            ai_msg = template[1].format(concept=random.choice(topics))
        elif "{concept}" in template[0] or "{concept}" in template[1]:
            concepts = ["Transformer", "注意力机制", "梯度下降", "反向传播"]
            concept = random.choice(concepts)
            human_msg = template[0].format(concept=concept)
            ai_msg = template[1].format(concept=concept)
        else:
            human_msg = template[0]
            ai_msg = template[1]
        
        return human_msg, ai_msg
    
    def _check_early_reference(
        self,
        human_msg: str,
        current_turn: int,
        previous_conversations: List[ConversationExample],
    ) -> Tuple[bool, Optional[int]]:
        """检查是否引用了早期信息"""
        # 简单的引用检测：检查消息中是否包含代词或提及之前的内容
        # 实际应用中可以使用更复杂的 NLP 方法
        
        # 如果对话少于5轮，不太可能引用早期信息
        if current_turn <= 5:
            return False, None
        
        # 检查是否包含引用性词汇
        reference_keywords = ["之前", "刚才", "之前提到", "还记得", "之前说的"]
        
        for keyword in reference_keywords:
            if keyword in human_msg:
                # 假设引用最近3轮内的某个关键信息
                ref_turn = max(1, current_turn - random.randint(1, 3))
                return True, ref_turn
        
        return False, None
    
    def _generate_test_questions(
        self,
        conversations: List[ConversationExample],
        key_info_map: Dict[int, str],
        total_turns: int,
    ) -> List[Dict[str, Any]]:
        """生成测试问题"""
        test_questions = []
        
        # 为每个关键信息生成测试问题
        for turn_num, key_info in key_info_map.items():
            # 根据关键信息类型生成问题
            if "姓名" in key_info or "名字" in key_info:
                question = "我的名字是什么？"
            elif "城市" in key_info or "来自" in key_info:
                question = "我来自哪个城市？"
            elif "年龄" in key_info:
                question = "我今年多大？"
            elif "职业" in key_info or "工作" in key_info:
                question = "我的职业是什么？"
            elif "编程语言" in key_info or "语言" in key_info:
                question = "我喜欢用什么编程语言？"
            elif "框架" in key_info:
                question = "我主要使用哪个框架？"
            elif "颜色" in key_info:
                question = "我最喜欢的颜色是什么？"
            else:
                question = f"关于{key_info}，你能告诉我什么？"
            
            test_questions.append({
                "question": question,
                "expected_key_info": key_info,
                "key_info_turn": turn_num,
                "test_after_turn": max(turn_num + 5, total_turns),  # 在关键信息出现后5轮测试
            })
        
        return test_questions
    
    def generate_standard_test_suite(self) -> Dict[str, TestDataset]:
        """
        生成标准测试套件
        
        Returns:
            包含不同长度对话的数据集字典
        """
        test_suites = {}
        
        # 10轮对话（短对话）
        test_suites["short_10"] = self.generate_long_conversation(
            total_turns=10,
            key_info_turns=[2, 5],
        )
        
        # 30轮对话（中等对话）
        test_suites["medium_30"] = self.generate_long_conversation(
            total_turns=30,
            key_info_turns=[3, 10, 20],
        )
        
        # 50轮对话（长对话）
        test_suites["long_50"] = self.generate_long_conversation(
            total_turns=50,
            key_info_turns=[5, 15, 30],
        )
        
        # 100轮对话（超长对话）
        test_suites["very_long_100"] = self.generate_long_conversation(
            total_turns=100,
            key_info_turns=[5, 20, 50, 80],
        )
        
        logger.info(f"生成标准测试套件: {len(test_suites)} 个数据集")
        
        return test_suites
