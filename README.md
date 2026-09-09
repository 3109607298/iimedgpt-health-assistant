# QwenMed — Qwen3-8B 医疗健康助手

一个可展示、也可连接私有 Qwen3-8B 服务的网页端。页面不把模型密钥发给浏览器，而是由 `server.py` 转发到 OpenAI 兼容的模型接口。

## 启动

1. 先启动你的 Qwen3-8B 推理服务。工作区现有约定是 `http://127.0.0.1:8000/v1`；请确认接口能访问 `GET /v1/models` 和 `POST /v1/chat/completions`。
2. 按实际服务填写环境变量（PowerShell 示例）：

```powershell
$env:QWEN_API_BASE = "http://127.0.0.1:8000/v1"
$env:QWEN_MODEL = "qwen3-8b"
$env:QWEN_API_KEY = "EMPTY"
python server.py
```

3. 浏览器打开 `http://127.0.0.1:5173`。

如果服务端使用其他模型名，请将 `QWEN_MODEL` 改成 `GET /v1/models` 返回的 `id`。`.env.example` 仅作配置参考，标准库版服务不会自动读取 `.env`，以避免引入额外依赖。

## 模型服务示例（vLLM）

```bash
vllm serve /path/to/Qwen3-8B --served-model-name qwen3-8b --host 0.0.0.0 --port 8000
```

如果模型位于 `/home/user/gptdata/CZH/model/Qwen/Qwen3-8B`，可在 GPU 服务器上执行项目中的 `start_qwen3_vllm.sh`：

```bash
chmod +x start_qwen3_vllm.sh
./start_qwen3_vllm.sh
curl http://127.0.0.1:8000/v1/models
```

网页运行在另一台机器时，请将系统设置中的服务地址改为 `http://<GPU服务器IP>:8000/v1`。确保服务器防火墙/安全组允许该端口访问；如仅限内网访问，请通过内网 IP 或 SSH 隧道连接。

医疗内容只用于健康科普与辅助参考，不替代医疗机构或专业医生的诊断、治疗与处方。
