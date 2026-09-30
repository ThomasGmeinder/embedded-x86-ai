# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""
测试情感分析系统
无需 Reachy Mini 仿真器即可运行
"""

import random
import re
from typing import Dict, Tuple


class EmotionAnalyzer:
    """情感分析器 - 分析文本情感并映射到机器人动作"""

    def __init__(self):
        # 情感关键词库
        self.emotion_patterns = {
            "positive": {
                "keywords": [
                    "开心",
                    "快乐",
                    "愉快",
                    "高兴",
                    "喜欢",
                    "爱",
                    "棒",
                    "好",
                    "优秀",
                    "完美",
                    "太棒了",
                    "谢谢",
                    "感谢",
                    "欢迎",
                    "帮助",
                    "支持",
                    "合作",
                    "理解",
                    "加油",
                    "努力",
                    "坚持",
                    "相信",
                    "一定",
                    "很棒",
                    "进步",
                    "成长",
                    "哇",
                    "天啊",
                    "真的吗",
                    "不可思议",
                    "惊人",
                    "太神奇了",
                ],
                "emojis": ["😊", "😄", "😍", "👍", "🥰", "😎", "🎉", "❤️"],
            },
            "negative": {
                "keywords": [
                    "伤心",
                    "难过",
                    "悲伤",
                    "失望",
                    "遗憾",
                    "抱歉",
                    "对不起",
                    "可惜",
                    "生气",
                    "愤怒",
                    "讨厌",
                    "烦人",
                    "可恶",
                    "该死",
                    "恨",
                    "担心",
                    "忧虑",
                    "害怕",
                    "恐惧",
                    "紧张",
                    "不安",
                    "焦虑",
                    "不",
                    "不能",
                    "不会",
                    "没有",
                    "不行",
                    "错误",
                    "失败",
                    "不对",
                ],
                "emojis": ["😢", "😭", "😡", "👎", "😔", "😞", "😤", "💔"],
            },
            "question": {
                "keywords": [
                    "吗",
                    "？",
                    "?",
                    "为什么",
                    "如何",
                    "怎样",
                    "什么时候",
                    "哪里",
                ],
                "patterns": [r"\?", r"？", r"why", r"how", r"what", r"when", r"where"],
            },
            "thinking": {
                "keywords": ["思考", "考虑", "分析", "研究", "探讨", "讨论"],
                "patterns": [r"我觉得", r"我认为", r"我想", r"可能是", r"大概"],
            },
        }

        # 情感强度判断
        self.intensity_indicators = {
            "high": ["非常", "特别", "极其", "超级", "极端", "太.+了", "！!{2,}"],
            "medium": ["很", "比较", "相当", "挺", "!"],
            "low": ["稍微", "一点", "有些"],
        }

    def analyze_emotion(self, text: str) -> Tuple[str, str]:
        """
        分析文本情感

        返回: (情感类型, 强度)
        情感类型: 'positive', 'negative', 'question', 'thinking', 'neutral'
        强度: 'high', 'medium', 'low'
        """
        text_lower = text.lower()

        # 计算各种情感的得分
        scores = {}
        for emotion_type, patterns in self.emotion_patterns.items():
            score = 0

            # 关键词匹配
            if "keywords" in patterns:
                for keyword in patterns["keywords"]:
                    if keyword in text_lower:
                        score += 1

            # 表情符号匹配
            if "emojis" in patterns:
                for emoji in patterns["emojis"]:
                    if emoji in text:
                        score += 2  # 表情符号权重更高

            # 正则表达式匹配
            if "patterns" in patterns:
                for pattern in patterns["patterns"]:
                    if re.search(pattern, text_lower):
                        score += 1

            scores[emotion_type] = score

        # 确定主导情感
        if max(scores.values()) == 0:
            emotion_type = "neutral"
        else:
            emotion_type = max(scores, key=scores.get)

        # 确定情感强度
        intensity = "low"
        text_lower = text.lower()
        for level, indicators in self.intensity_indicators.items():
            for indicator in indicators:
                if re.search(indicator, text_lower):
                    intensity = level
                    break

        return emotion_type, intensity

    def get_emotion_description(self, emotion_type: str, intensity: str) -> str:
        """获取情感描述"""
        descriptions = {
            "positive": {
                "high": "非常高兴 😄",
                "medium": "高兴 😊",
                "low": "轻微高兴 🙂",
            },
            "negative": {
                "high": "非常难过 😢",
                "medium": "难过 😔",
                "low": "轻微不开心 🙁",
            },
            "question": {
                "high": "强烈疑问 🤔❓",
                "medium": "疑问 🤔",
                "low": "轻微疑问 ⁉️",
            },
            "thinking": {
                "high": "深度思考 🤔💭",
                "medium": "思考 🤔",
                "low": "轻微思考 💭",
            },
            "neutral": {"high": "平静 😶", "medium": "中性 😐", "low": "轻微中性 🙂"},
        }

        return descriptions.get(emotion_type, {}).get(
            intensity, f"{emotion_type} ({intensity})"
        )

    def suggest_robot_actions(self, emotion_type: str, intensity: str) -> Dict:
        """建议机器人动作"""
        actions_suggestions = {
            "positive": {
                "high": {
                    "actions": [
                        "快速点头",
                        "天线活泼摆动",
                        "轻微摇头（开心）",
                        "快速复位",
                    ],
                    "amplitude": 0.8,
                    "speed": "快",
                    "duration": "短（0.5秒每个动作）",
                },
                "medium": {
                    "actions": ["中等幅度点头", "天线中等摆动", "缓慢摇头"],
                    "amplitude": 0.5,
                    "speed": "中等",
                    "duration": "中等（1秒每个动作）",
                },
                "low": {
                    "actions": ["轻微点头", "天线轻微摆动", "缓慢复位"],
                    "amplitude": 0.3,
                    "speed": "慢",
                    "duration": "长（1.5秒每个动作）",
                },
            },
            "negative": {
                "high": {
                    "actions": [
                        "低头（悲伤）",
                        "缓慢沉重摇头",
                        "天线低垂",
                        "长时间保持",
                    ],
                    "amplitude": 0.8,
                    "speed": "很慢",
                    "duration": "很长（2秒每个动作）",
                },
                "medium": {
                    "actions": ["中等低头", "缓慢摇头", "天线微垂"],
                    "amplitude": 0.5,
                    "speed": "慢",
                    "duration": "长（1.5秒每个动作）",
                },
                "low": {
                    "actions": ["轻微低头", "缓慢复位"],
                    "amplitude": 0.3,
                    "speed": "很慢",
                    "duration": "很长（2秒每个动作）",
                },
            },
            "question": {
                "high": {
                    "actions": ["头偏向一边（好奇）", "快速天线摆动", "快速复位"],
                    "amplitude": 0.8,
                    "speed": "快",
                    "duration": "短（0.5秒每个动作）",
                },
                "medium": {
                    "actions": ["头中等偏斜", "天线中等摆动", "中等复位"],
                    "amplitude": 0.5,
                    "speed": "中等",
                    "duration": "中等（1秒每个动作）",
                },
                "low": {
                    "actions": ["头轻微偏斜", "天线轻微摆动", "缓慢复位"],
                    "amplitude": 0.3,
                    "speed": "慢",
                    "duration": "长（1.5秒每个动作）",
                },
            },
            "thinking": {
                "high": {
                    "actions": ["缓慢深沉点头", "缓慢深沉摇头", "长时间保持"],
                    "amplitude": 0.8,
                    "speed": "很慢",
                    "duration": "很长（2秒每个动作）",
                },
                "medium": {
                    "actions": ["中等缓慢点头", "中等缓慢摇头", "缓慢复位"],
                    "amplitude": 0.5,
                    "speed": "慢",
                    "duration": "长（1.5秒每个动作）",
                },
                "low": {
                    "actions": ["轻微点头", "轻微摇头", "缓慢复位"],
                    "amplitude": 0.3,
                    "speed": "很慢",
                    "duration": "很长（2秒每个动作）",
                },
            },
            "neutral": {
                "high": {
                    "actions": ["轻微中性点头", "缓慢复位"],
                    "amplitude": 0.3,
                    "speed": "慢",
                    "duration": "长（1.5秒）",
                },
                "medium": {
                    "actions": ["极轻微点头", "缓慢复位"],
                    "amplitude": 0.2,
                    "speed": "很慢",
                    "duration": "很长（2秒）",
                },
                "low": {
                    "actions": ["几乎不动", "非常缓慢复位"],
                    "amplitude": 0.1,
                    "speed": "极慢",
                    "duration": "很长（2.5秒）",
                },
            },
        }

        return actions_suggestions.get(emotion_type, {}).get(intensity, {})


def run_emotion_analysis_demo(interactive=False):
    print("=" * 70)
    print("🤖 情感分析系统测试")
    print("=" * 70)

    analyzer = EmotionAnalyzer()

    # 测试用例
    test_cases = [
        "我非常开心！今天天气真好！😊",
        "我感到很难过... 😢",
        "这个问题该怎么解决？",
        "让我思考一下...",
        "好的，明白了。",
        "太棒了！你做得非常好！🎉",
        "我真的很生气！😡",
        "你觉得这个方案怎么样？🤔",
        "我只是有点担心...",
        "谢谢你的帮助！❤️",
    ]

    print("\n测试用例分析结果:")
    print("-" * 70)

    for i, text in enumerate(test_cases, 1):
        emotion_type, intensity = analyzer.analyze_emotion(text)
        description = analyzer.get_emotion_description(emotion_type, intensity)
        actions = analyzer.suggest_robot_actions(emotion_type, intensity)

        print(f"\n{i}. 文本: {text}")
        print(f"   情感分析: {description}")
        print(f"   动作建议:")
        if actions:
            print(f"     - 动作: {', '.join(actions.get('actions', []))}")
            print(f"     - 幅度: {actions.get('amplitude', 0)}")
            print(f"     - 速度: {actions.get('speed', 'N/A')}")
            print(f"     - 持续时间: {actions.get('duration', 'N/A')}")
        else:
            print(f"     - 无特定动作建议")

    print("\n" + "=" * 70)
    print("✅ 情感分析测试完成！")
    print("=" * 70)

    if not interactive:
        return

    # 交互式测试
    print("\n🎮 交互式测试 (输入 'quit' 退出)")
    print("-" * 70)

    while True:
        try:
            user_input = input("\n请输入测试文本: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 退出测试")
                break

            if not user_input:
                continue

            emotion_type, intensity = analyzer.analyze_emotion(user_input)
            description = analyzer.get_emotion_description(emotion_type, intensity)
            actions = analyzer.suggest_robot_actions(emotion_type, intensity)

            print(f"\n📊 分析结果:")
            print(f"   情感: {description}")
            print(f"   动作建议:")
            if actions:
                print(f"     - 动作: {', '.join(actions.get('actions', []))}")
                print(f"     - 幅度: {actions.get('amplitude', 0)}")
                print(f"     - 速度: {actions.get('speed', 'N/A')}")
                print(f"     - 持续时间: {actions.get('duration', 'N/A')}")
            else:
                print(f"     - 无特定动作建议")

            # 显示对应情感的表情符号
            print(f"\n🎭 对应表情:")
            emotion_emojis = {
                "positive": "😊😄😍👍",
                "negative": "😢😭😡👎",
                "question": "🤔❓⁉️",
                "thinking": "🤔💭🧐",
                "neutral": "😶🙂😐",
            }
            print(f"    {emotion_emojis.get(emotion_type, '😶')}")

        except KeyboardInterrupt:
            print("\n\n👋 中断测试")
            break
        except Exception as e:
            print(f"\n⚠️ 错误: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试情感分析系统")
    parser.add_argument(
        "--demo", action="store_true", help="运行非交互式演示（默认，不等待用户输入）"
    )
    parser.add_argument("--interactive", action="store_true", help="启用交互式测试模式")
    args = parser.parse_args()

    run_emotion_analysis_demo(interactive=args.interactive)


if __name__ == "__main__":
    main()
