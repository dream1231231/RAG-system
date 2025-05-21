# modules/dialogue.py
class DialogueManager:
    """对话历史管理"""

    def __init__(self, max_history=3):
        self.max_history = max_history
        self.history = []  # 格式: [(query, response)]

    def update(self, query: str, response: str):
        """更新历史记录"""
        self.history.append((query, response))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        """生成对话上下文"""
        return "\n".join(
            [f"第{i + 1}轮问：{q}\n第{i + 1}轮答：{a[:100]}..."
             for i, (q, a) in enumerate(self.history)]
        )

    def refine_query(self, current_query: str) -> str:
        """迭代优化查询"""
        if not self.history:
            return current_query
        # 提取历史中的关键实体（示例：简单拼接）
        return f"{self.get_context()}\n当前问题：{current_query}"