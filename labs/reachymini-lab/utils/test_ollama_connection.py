# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""
测试 Ollama 连接
"""


def check_ollama_connection(ollama_url="http://localhost:11434"):
    """测试 Ollama 连接"""
    import requests
    import socket
    import json

    print("=" * 60)
    print("🔧 测试 Ollama 连接")
    print("=" * 60)

    # 测试 1: 直接 HTTP 请求
    print("\n1. 测试 HTTP API 连接...")
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print(f"✅ HTTP API 连接成功 (状态码: {response.status_code})")

            # 显示可用模型
            data = response.json()
            models = data.get("models", [])
            if models:
                print(f"✅ 可用模型:")
                for model in models:
                    print(f"   - {model.get('name', '未知模型')}")
            else:
                print("⚠️  没有找到模型")
                print("   请运行: ollama pull qwen3.5:0.8b")
        else:
            print(f"❌ HTTP API 连接失败 (状态码: {response.status_code})")
            print(f"   响应: {response.text}")
    except Exception as e:
        print(f"❌ HTTP API 连接错误: {type(e).__name__}")
        print(f"   详细: {str(e)}")

    # 测试 2: OpenAI SDK 连接
    print("\n2. 测试 OpenAI SDK 连接...")
    try:
        # Lazy import to keep module importable in environments without openai package.
        from openai import OpenAI

        client = OpenAI(base_url=f"{ollama_url}/v1", api_key="no-key-needed")

        # 测试简单请求
        response = client.chat.completions.create(
            model="qwen3.5:0.8b",
            messages=[
                {"role": "system", "content": "你是一个测试助手"},
                {"role": "user", "content": "你好"},
            ],
            max_tokens=10,
        )

        if response.choices and response.choices[0].message.content:
            print(f"✅ OpenAI SDK 连接成功")
            print(f"   测试回复: {response.choices[0].message.content}")
        else:
            print(f"⚠️  OpenAI SDK 连接异常: 无回复内容")

    except Exception as e:
        print(f"❌ OpenAI SDK 连接错误: {type(e).__name__}")
        print(f"   详细: {str(e)}")

    # 测试 3: 流式响应
    print("\n3. 测试流式响应...")
    try:
        # Lazy import to keep module importable in environments without openai package.
        from openai import OpenAI

        client = OpenAI(base_url=f"{ollama_url}/v1", api_key="no-key-needed")

        stream = client.chat.completions.create(
            model="qwen3.5:0.8b",
            messages=[
                {"role": "system", "content": "你是一个测试助手"},
                {"role": "user", "content": "测试流式响应"},
            ],
            stream=True,
            max_tokens=20,
        )

        print("✅ 流式响应测试开始...")
        print("   响应: ", end="", flush=True)

        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print()  # 换行

        if full_response:
            print(f"✅ 流式响应成功 (长度: {len(full_response)} 字符)")
        else:
            print("⚠️  流式响应无内容")

    except Exception as e:
        print(f"❌ 流式响应错误: {type(e).__name__}")
        print(f"   详细: {str(e)}")

    # 总结
    print("\n" + "=" * 60)
    print("📋 连接测试总结")
    print("=" * 60)

    # 检查系统状态
    print("\n系统状态检查:")

    # 检查端口
    try:
        host = ollama_url.split("//")[1].split(":")[0]
        port = int(ollama_url.split(":")[-1].split("/")[0])

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))

        if result == 0:
            print(f"✅ 端口 {port} 可达")
        else:
            print(f"❌ 端口 {port} 不可达")
            print(f"   请检查: 1) Ollama 是否运行 2) 防火墙设置")

        sock.close()
    except Exception as e:
        print(f"⚠️  端口检查错误: {e}")

    print("\n建议:")
    print("1. 如果所有测试都失败，请确保 Ollama 正在运行")
    print("2. 运行: ollama serve")
    print("3. 拉取模型: ollama pull qwen3.5:0.8b")
    print("4. 检查防火墙: sudo lsof -i :11434")
    print("5. 尝试其他端口: 修改 ollama_url 参数")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="测试 Ollama 连接")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--model", default="qwen3.5:0.8b", help="测试模型")

    args = parser.parse_args()

    check_ollama_connection(args.url)


if __name__ == "__main__":
    main()
