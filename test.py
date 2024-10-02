import os
import subprocess
import time
import webbrowser
from torch.utils.tensorboard import SummaryWriter

# 设置 TensorBoard 日志路径
log_dir = "runs/experiment1"

# 启动 TensorBoard 服务器
def start_tensorboard(log_dir):
    # 确保路径存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 启动 TensorBoard 进程
    tensorboard_command = ["tensorboard", "--logdir", log_dir, "--port", "6006", "--reload_interval", "5"]
    tb_process = subprocess.Popen(tensorboard_command)
    
    # 等待 TensorBoard 启动
    time.sleep(5)
    
    # 打开默认浏览器访问 TensorBoard
    try:
        # 尝试使用 Windows 默认浏览器
        subprocess.run(["explorer.exe", "http://localhost:6006"])
    except webbrowser.Error:
        # 如果失败，则尝试使用系统默认浏览器
        webbrowser.open("http://localhost:6006")
    
    return tb_process

# 创建 SummaryWriter 实例
writer = SummaryWriter(log_dir=log_dir)

# 启动 TensorBoard
tb_process = start_tensorboard(log_dir)

# 模拟训练过程并记录标量
for epoch in range(1, 1111):
    # 模拟损失值
    loss = 0.5 / epoch
    writer.add_scalar('Training Loss', loss, epoch)
    print(f"Epoch {epoch}, Loss: {loss}")
    time.sleep(1)  # 模拟训练延迟

# 关闭 SummaryWriter
writer.close()

# 关闭 TensorBoard 进程
tb_process.terminate()

print("Training finished. TensorBoard is still running.")
try:
    input("Press Enter to stop TensorBoard and exit...")
finally:
    # 关闭 TensorBoard 进程
    tb_process.terminate()
