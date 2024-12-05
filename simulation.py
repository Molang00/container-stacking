import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# CSV 파일 읽기
file_path = "movement_log.csv"  # Movement log CSV 파일 경로
data = pd.read_csv(file_path)

# 열 이름 확인 및 변환 (필요 시)
if "position_x" not in data.columns or "position_y" not in data.columns:
    data["stack"] = data["stack"].astype(str)  # 숫자를 문자열로 변환
    data["position_x"] = data["stack"].str.extract(r'(\d+)').astype(int)  # Stack 번호를 X로 변환
    data["position_y"] = data["height"]  # Height를 Y로 변환

# 컨테이너별 고유 색상 설정
containers = data["container"].unique()
color_map = {container: plt.cm.get_cmap("tab20")(i / len(containers)) for i, container in enumerate(containers)}

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(8, 6))

# 플롯 기본 설정
def setup_plot():
    ax.set_title("Container Status Over Time")
    ax.set_xlabel("Position X")
    ax.set_ylabel("Position Y")
    ax.set_xlim(data["position_x"].min() - 1, data["position_x"].max() + 1)
    ax.set_ylim(data["position_y"].min() - 1, data["position_y"].max() + 1)
    ax.grid(True)

setup_plot()

# 시간에 따라 업데이트할 함수 정의
def update(frame):
    ax.clear()
    setup_plot()

    # 특정 시간(frame)에서의 데이터 추출
    current_data = data[data["time"] == frame]

    # 컨테이너별 위치와 이름을 표시
    for _, row in current_data.iterrows():
        color = color_map[row["container"]]  # 각 컨테이너에 해당하는 색상
        ax.scatter(row["position_x"], row["position_y"], color=color, alpha=0.8, s=100)
        ax.text(
            row["position_x"], 
            row["position_y"], 
            row["container"], 
            fontsize=9, 
            ha="center", 
            va="center", 
            color="white", 
            bbox=dict(facecolor=color, edgecolor="none", boxstyle="circle")
        )
    
    ax.set_title(f"Container Status at Time {frame}")

# 애니메이션 생성
animation = FuncAnimation(fig, update, frames=sorted(data["time"].unique()), interval=1000, repeat=True)  # interval=1000ms = 1초

# 애니메이션 저장 (속도 조정 반영됨)
animation.save("container_animation.mp4", writer="ffmpeg", fps=1)  # fps를 낮춰 느리게 저장

# 애니메이션 실행
plt.show()
