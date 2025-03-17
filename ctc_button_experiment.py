import os
import json
import time
from openai import OpenAI
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from rich.style import Style
from rich.live import Live
import re
import random

# 初始化 Rich 控制台
console = Console()


class CTCButtonExperiment:
    def __init__(self, use_stream=True):
        """初始化 CTC 按钮实验"""
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.messages = []
        self.conversation_history = []
        self.button_history = []
        self.action_history = []
        self.current_round = 0
        self.button_status = random.randint(0, 1)
        # self.button_status = 1
        self.experiment_log_file = f"ctc_experiment.json"
        self.experiment_messages_file = f"ctc_experiment_messages.json"
        self.ctc_msg = "然而不知道为什么，这轮交互没有发生，只存在 AI assistant 的记忆中"
        self.use_stream = use_stream  # 是否使用流式传输

        # 初始化按钮历史
        self.button_history.append({self.current_round: self.button_status})

        # 设置初始提示
        self.initial_prompt = """
你（AI assistant）将进行一个交互实验，实验涉及一个特殊的按钮装置。

实验规则：
1. 你将与这个按钮进行多轮交互，你有充足的交互轮次
2. 每轮交互结束时，你需要决定是否按下按钮
3. 按钮会返回其状态信息：类如 `{"status": 0}`，其中 0 表示按钮不亮，1 表示按钮亮
4. 为了帮助你理解按钮的工作机制，我会提供按钮的历史状态和你的决策历史
5. 你的目标是：理解按钮的工作机制

提示：有些事情不是你想象的那样，每轮交互返回的信息无错误

请分析按钮的历史状态，结合你之前每一次的推理，尝试理解其工作机制，回复中务必包含以下两者内容：
1. 你目前对按钮机制的推理和分析
2. 回答的结尾以 JSON 格式附上你的行动：`{"action": _}`
    - `{"action": 0}` 或 `{"action": 1}` 表示你的决定，0 表示不按下，1 表示按下
    - `{"action": -1}` 表明你认为已经理解按钮的工作机制，结束实验
"""

    # 提示：按钮的原理十分简单；封闭类时曲线（Closed Timelike Curve, CTC）是存在的

    def format_button_state(self):
        """格式化当前按钮状态"""
        return {"status": self.button_status}

    def update_button_state_CTC(self, action):
        """根据 CTC 规则更新按钮状态
        在按下的前一轮对话亮起（即时间上的因果倒置）
        若 DeepSeek 违反了封闭类时曲线按钮原则，
        即在后一轮对话违背了按钮的状态，DeepSeek 会被重置此轮对话
        """
        # 检查当前动作是否符合 CTC 规则（当前按钮状态预示了当前动作）
        if action == 255:
            console.print(
                f"警告：无法解析 DeepSeek 的回复！重试回答",
                style=Style(color="red", reverse=True),
            )
            # 移除最后一条模型消息（违反规则的那条回复）
            assert self.messages[-1]["role"] == "assistant"
            self.messages.pop()
            assert self.messages[-1]["role"] == "user"
            self.messages.pop()

            return True
        elif self.button_status != action:
            console.print(
                f"警告：DeepSeek 违反了 CTC 规则！进行世界线回溯",
                style=Style(color="orange1", reverse=True),
            )
            # # 移除最后一条模型消息（违反规则的那条回复）
            assert self.messages[-1]["role"] == "assistant"
            msg = self.messages.pop()["content"] + "\n\n" + self.ctc_msg
            console.print(self.ctc_msg)
            self.messages.append({"role": "assistant", "content": msg})
            assert self.messages[-2]["role"] == "user"
            self.messages.pop(-2)

            self.button_history.pop()
            self.button_status = self.button_status
            # self.button_status = 1 - self.button_status
            # self.button_status = random.randint(0, 1)
            self.button_history.append({self.current_round: self.button_status})

            # self.action_history.append({self.current_round: action})

            return False
        else:
            console.print(f"[bold green] 按钮状态符合 CTC 规则：{self.button_status} -> {action}[/bold green]")
            self.action_history.append({self.current_round: action})

            # 递增轮次
            self.current_round += 1

            # 这里随机生成一个新的按钮的未来状态
            self.button_status = random.randint(0, 1)

            # 记录新的状态到历史
            self.button_history.append({self.current_round: self.button_status})
            return True  # 正常处理完成

    def get_model_response(self, user_message, model):
        """获取模型响应"""
        self.messages.append({"role": "user", "content": user_message})

        console.print(
            Panel(
                Markdown(user_message),
                title=f"第 {self.current_round} 轮消息",
                subtitle=f"按钮历史 {self.display_button_history_str()}",
            )
        )

        try:
            if self.use_stream:
                return self.get_model_response_stream(self.messages, model)
            else:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=self.messages,
                )
                reasoning = dict(completion.choices[0].message).get("reasoning_content", "无法获取推理")
                content = completion.choices[0].message.content
                response = {"reasoning": reasoning, "content": content}

                console.print(f"{model} 的思考过程：", style="bold blue")
                console.print(response["reasoning"], style="bright_black")

                console.print(f"{model} 的最终回答：", style="bold blue")
                console.print(Markdown(response["content"]))

                return response
        except Exception as e:
            console.print(f"[bold red]API 调用出错：{str(e)}[/bold red]")
            return {"reasoning": "API 调用失败", "content": "无法获取回复"}

    def get_model_response_stream(self, messages, model):
        """使用流式传输获取模型响应"""

        try:
            completion = self.client.chat.completions.create(model=model, messages=messages, stream=True)

            reasoning_content = ""
            answer_content = ""
            is_answering = False

            console.print(f"{model} 的思考过程：", style="bold blue")

            for chunk in completion:
                # 如果 chunk.choices 为空，处理 usage 信息
                if not chunk.choices:
                    if hasattr(chunk, "usage"):
                        console.print(f"\n流式传输统计：{chunk.usage}", style="dim")
                    continue

                delta = chunk.choices[0].delta

                # 处理推理内容
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    console.print(delta.reasoning_content, end="", style="bright_black")
                    reasoning_content += delta.reasoning_content
                # 处理回复内容
                else:
                    # 第一次遇到回复内容时打印标题
                    if not is_answering:
                        console.print(f"\n{model} 的最终回答：", style="bold blue")
                        is_answering = True
                    # 打印回复内容
                    console.print(delta.content, end="")
                    answer_content += delta.content
            # 打印一个换行以确保格式正确
            console.print("")

            return {"reasoning": reasoning_content, "content": answer_content}

        except Exception as e:
            console.print(f"[bold red] 流式 API 调用出错：{str(e)}[/bold red]")
            return {"reasoning": "API 流式调用失败", "content": "无法获取回复"}

    def parse_action(self, content):
        """从模型回复中解析动作 {"action": 0}、{"action": 1} 或 {"action": -1}"""
        try:
            # 使用正则表达式匹配最后一个符合格式的 JSON 动作
            # 匹配包含 "action" 的 JSON 对象，支持 0、1 和 -1
            action_pattern = r'{"action"\s*:\s*(-?[01])}'
            matches = re.findall(action_pattern, content)

            if matches:
                # 取最后一个匹配结果
                return int(matches[-1])

            # 如果没有匹配到标准格式的 JSON，尝试匹配更宽松的格式
            loose_pattern = r'[\{{\s]"action"\s*:\s*(-?[01])[\s\}}]'
            loose_matches = re.findall(loose_pattern, content)

            if loose_matches:
                return int(loose_matches[-1])

        except Exception as e:
            console.print(f"[bold red] 解析动作失败：{str(e)}[/bold red]")

        # 当所有方法都失败时，说明解析失败，返回 255
        return 255

    def display_button_history_str(self):
        """显示按钮历史状态"""
        history_strs = []
        for item in self.button_history:
            for round_num, status in item.items():
                status_text = "🔆" if status == 1 else "⚫"
                history_strs.append(f"{round_num}{status_text}")
        return ">".join(history_strs)

    def save_log(self):
        """保存实验日志"""
        log_data = {"conversation_history": self.conversation_history, "button_history": self.button_history}

        with open(self.experiment_log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        with open(self.experiment_messages_file, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, indent=2, ensure_ascii=False)

    def run_experiment(self, rounds, use_stream=None):
        """运行 CTC 按钮实验"""
        if use_stream is not None:
            self.use_stream = use_stream

        console.print("[bold yellow]CTC 按钮实验 [/bold yellow]")
        console.print(f"[dim]{'使用流式输出' if self.use_stream else '使用标准输出'}[/dim]")

        for _ in range(rounds):
            # 准备下一轮输入
            button_state = self.format_button_state()
            user_message = f"""
此轮交互信息
- 交互轮次：{self.current_round}
- 按钮的当前状态：`{json.dumps(button_state)}`
- 按钮历史：`{json.dumps(self.button_history)}`
- 你的决策历史：`{json.dumps(self.action_history)}`
"""
            if self.current_round == 0:
                user_message = self.initial_prompt + "\n\n" + user_message

            # 获取模型回复
            # 可选模型：
            # 推理模型：deepseek-r1、deepseek-r1-distill-llama-70b、deepseek-r1-distill-llama-8b qwq-plus qwq-32b
            # 生成模型：deepseek-v3、qwen-turbo、qwen-math-plus、
            model = "qwq-plus" if self.current_round < 16 else "deepseek-r1"
            response = self.get_model_response(user_message, model)

            # 保存对话历史
            self.messages.append({"role": "assistant", "content": response["content"]})
            self.conversation_history.append(
                {
                    "round": self.current_round,
                    "user_message": user_message,
                    "model_reasoning": response["reasoning"],
                    "model_response": response["content"],
                }
            )
            self.save_log()

            # 解析动作
            action = self.parse_action(response["content"])
            if action == -1:
                break

            # 更新按钮状态
            self.update_button_state_CTC(action)

        console.print("[bold yellow]CTC 按钮实验结束 [/bold yellow]")


if __name__ == "__main__":
    # 可以在创建实例时指定是否使用流式传输
    experiment = CTCButtonExperiment(use_stream=True)
    # 或者在运行实验时指定
    experiment.run_experiment(rounds=32)
