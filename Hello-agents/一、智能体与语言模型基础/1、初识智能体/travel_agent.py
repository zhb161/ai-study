"""
智能旅行助手 - 基于 Thought-Action-Observation 范式的智能体实现

本示例演示如何从零开始构建一个能处理分步任务的智能旅行助手。
用户任务："你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"

该智能体会展现清晰的逻辑规划能力：
1. 先调用天气查询工具获取当前天气
2. 将天气结果作为下一步的依据
3. 调用景点推荐工具，得出最终建议

  运行命令：

  pip install requests tavily-python openai
  python travel_agent.py
"""
import random
import os
import re
import requests
import json
from openai import OpenAI
from tavily import TavilyClient


def load_env_file(filepath=".env"):
    """
    手动加载 .env 文件中的环境变量。
    不需要安装 python-dotenv。
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith("#"):
                    continue
                # 解析 KEY=VALUE
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # 只设置未存在的环境变量
                    if key and key not in os.environ:
                        os.environ[key] = value
        print(f"✅ 已加载 {filepath} 文件中的配置")


# 加载 .env 文件配置
load_env_file()

# =============================================================================
# 1. 指令模板 (Prompt Engineering)
# =============================================================================

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动，每次回复只输出一对Thought-Action：
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成:
当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `finish(answer="...")` 来输出最终答案。

请开始吧！
"""


# =============================================================================
# 2. 工具函数定义
# =============================================================================

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    
    Args:
        city: 城市名称（中文或英文）
    
    Returns:
        格式化的天气描述字符串
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"
    """模拟天气查询，用于测试"""

    # 模拟网络延迟
    # time.sleep(0.5)

    # 模拟几种天气状况
    weathers = [
        ("晴", 25), ("多云", 22), ("阴", 20),
        ("小雨", 18), ("雷阵雨", 19), ("霾", 21)
    ]

    # 根据城市名哈希，让同一城市结果固定
    idx = hash(city) % len(weathers)
    desc, temp = weathers[idx]

    return f"{city}当前天气：{desc}，气温{temp}°C（模拟数据）"
    # try:
    #     # 发起网络请求
    #     response = requests.get(url)
    #     # 检查响应状态码是否为200 (成功)
    #     response.raise_for_status()
    #     # 解析返回的JSON数据
    #     data = response.json()
    #
    #     # 提取当前天气状况
    #     current_condition = data['current_condition'][0]
    #     weather_desc = current_condition['weatherDesc'][0]['value']
    #     temp_c = current_condition['temp_C']
    #
    #     # 格式化成自然语言返回
    #     return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
    #
    # except requests.exceptions.RequestException as e:
    #     # 处理网络错误
    #     return f"错误:查询天气时遇到网络问题 - {e}"
    # except (KeyError, IndexError) as e:
    #     # 处理数据解析错误
    #     return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    
    Args:
        city: 城市名称
        weather: 天气状况描述
    
    Returns:
        景点推荐信息
    """
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    # 3. 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


# 将所有工具函数放入一部字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


# =============================================================================
# 3. LLM 客户端实现
# =============================================================================

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    
    支持 OpenAI、Azure、Ollama、vLLM 等服务。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        """
        初始化LLM客户端。
        
        Args:
            model: 模型ID（如 "gpt-4o", "qwen2.5:14b" 等）
            api_key: API密钥
            base_url: API基础URL
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        调用LLM API来生成回应。
        
        Args:
            prompt: 用户提示（包含历史对话）
            system_prompt: 系统提示（定义智能体行为）
        
        Returns:
            LLM生成的文本响应
        """
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


# =============================================================================
# 4. 主执行循环
# =============================================================================

def run_agent(
        user_prompt: str,
        llm_client: OpenAICompatibleClient,
        max_iterations: int = 5
) -> str:
    """
    运行智能体主循环，处理用户请求。
    
    Args:
        user_prompt: 用户输入的任务描述
        llm_client: LLM客户端实例
        max_iterations: 最大循环次数（防止无限循环）
    
    Returns:
        最终答案
    """
    prompt_history = [f"用户请求: {user_prompt}"]

    print(f"用户输入: {user_prompt}\n" + "=" * 40)

    for i in range(max_iterations):
        print(f"\n--- 循环 {i + 1} ---\n")

        # 4.1. 构建Prompt（包含历史对话）
        full_prompt = "\n".join(prompt_history)

        # 4.2. 调用LLM进行思考
        llm_output = llm_client.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)

        # 模型可能会输出多余的Thought-Action，需要截断只保留第一对
        match = re.search(
            r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
            llm_output,
            re.DOTALL
        )
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")

        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        # 4.3. 解析并执行行动
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            print("解析错误:模型输出中未找到 Action。")
            break

        action_str = action_match.group(1).strip()

        # 检查是否是完成任务
        if action_str.startswith("finish"):
            final_match = re.search(r'finish\(answer="(.*)"\)', action_str, re.DOTALL)
            if final_match:
                final_answer = final_match.group(1)
                print(f"\n{'=' * 40}")
                print(f"任务完成，最终答案: {final_answer}")
                return final_answer
            else:
                print("解析错误:无法解析 finish 参数。")
                break

        # 解析工具调用
        tool_match = re.search(r"(\w+)\((.*)\)", action_str, re.DOTALL)
        if not tool_match:
            print(f"解析错误:无法解析 Action 格式: {action_str}")
            break

        tool_name = tool_match.group(1)
        args_str = tool_match.group(2)

        # 解析关键字参数（格式：key="value"）
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        # 执行工具
        if tool_name in available_tools:
            print(f"执行工具: {tool_name}({kwargs})")
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        # 4.4. 记录观察结果
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "=" * 40)
        prompt_history.append(observation_str)

    else:
        print(f"\n达到最大循环次数({max_iterations})，任务未能在限定次数内完成。")

    return "任务未能完成"


# =============================================================================
# 5. 入口函数
# =============================================================================

def main():
    """
    主入口函数，演示智能旅行助手的完整工作流程。
    
    配置优先级（从高到低）：
    1. 环境变量（推荐，适合生产环境）
    2. .env 配置文件（推荐，适合开发环境）
    3. 代码中的默认值（仅适合快速测试）
    """

    # ========== 配置方式说明 ==========
    # 
    # 【方式1：环境变量】（推荐用于生产环境）
    #   在命令行设置：
    #   Windows CMD: set OPENAI_API_KEY=sk-xxxxx
    #   Windows PowerShell: $env:OPENAI_API_KEY="sk-xxxxx"
    #   Linux/Mac: export OPENAI_API_KEY=sk-xxxxx
    #
    # 【方式2：.env 配置文件】（推荐用于开发环境）
    #   在同目录下创建 .env 文件，内容如下：
    #   OPENAI_API_KEY=sk-xxxxx
    #   OPENAI_BASE_URL=https://api.openai.com/v1
    #   OPENAI_MODEL_ID=gpt-4o
    #   TAVILY_API_KEY=tvly-xxxxx
    #   然后安装: pip install python-dotenv
    #
    # 【方式3：直接修改下方默认值】（仅适合快速测试）
    #   将 YOUR_API_KEY 等占位符替换为实际值
    #
    # ================================

    # 从环境变量读取配置（如果未设置则使用默认值）
    API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "YOUR_BASE_URL")
    MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "YOUR_MODEL_ID")

    # 检查配置是否有效
    if API_KEY == "YOUR_API_KEY" or BASE_URL == "YOUR_BASE_URL" or MODEL_ID == "YOUR_MODEL_ID":
        print("\n" + "=" * 60)
        print("⚠️  配置错误：请先设置 API 配置信息！")
        print("=" * 60)
        print("\n请通过以下任一方式配置：\n")
        print("【方式1】环境变量（推荐）：")
        print("   Windows PowerShell:")
        print('   $env:OPENAI_API_KEY="sk-xxxxx"')
        print('   $env:OPENAI_BASE_URL="https://api.openai.com/v1"')
        print('   $env:OPENAI_MODEL_ID="gpt-4o"')
        print('   $env:TAVILY_API_KEY="tvly-xxxxx"')
        print()
        print("   Linux/Mac:")
        print('   export OPENAI_API_KEY=sk-xxxxx')
        print('   export OPENAI_BASE_URL=https://api.openai.com/v1')
        print('   export OPENAI_MODEL_ID=gpt-4o')
        print('   export TAVILY_API_KEY=tvly-xxxxx')
        print()
        print("【方式2】创建 .env 文件（推荐开发时使用）：")
        print("   1. 在同目录下创建 .env 文件")
        print("   2. 文件内容参考 .env.example")
        print("   3. pip install python-dotenv")
        print("=" * 60 + "\n")
        return

    # 检查Tavily API密钥
    if not os.environ.get("TAVILY_API_KEY"):
        print("\n⚠️  警告: 未设置 TAVILY_API_KEY 环境变量，景点推荐功能将无法使用。")
        print("   请设置环境变量：TAVILY_API_KEY=tvly-xxxxx\n")

    # 初始化LLM客户端
    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # --- 用户输入 ---
    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"

    # --- 运行智能体 ---
    result = run_agent(user_prompt, llm)

    return result


if __name__ == "__main__":
    main()
