import numpy as np
import os
import matplotlib.pyplot as plt

RING_NUM = 104
CRYSTAL_NUM_ONE_RING = 312

MICH_OFFSET = 6

# __michPath: mich path
# __view: 选择视图索引
def plotFigure(__michPath, __view = -1):
    isOneView = False   
    if __view >= 0: 
        isOneView = True

    # 加载数据
    prompt = np.fromfile("/home/ustc/Desktop/E180_20250826_WellCounter_PET4643_Slice4796/Data/source/prompt.pni", dtype=np.float32, offset=MICH_OFFSET).reshape([RING_NUM*RING_NUM, int(CRYSTAL_NUM_ONE_RING / 2), int(CRYSTAL_NUM_ONE_RING - 1)])
    delay = np.fromfile("/home/ustc/Desktop/E180_20250826_WellCounter_PET4643_Slice4796/Data/result/random_factor.pni", dtype=np.float32, offset=0).reshape([RING_NUM*RING_NUM, int(CRYSTAL_NUM_ONE_RING / 2), int(CRYSTAL_NUM_ONE_RING - 1)])
    scatter = np.fromfile("/home/ustc/Desktop/testE180Case/MichScatter_SSSTailFitting.bin", dtype=np.float32, offset=0).reshape([RING_NUM*RING_NUM, int(CRYSTAL_NUM_ONE_RING / 2), int(CRYSTAL_NUM_ONE_RING - 1)])

    # 对数据进行求和处理
    pr_sl = prompt.sum(axis=0)
    de_sl = delay.sum(axis=0)
    sc_sl = scatter.sum(axis=0)

    pr = pr_sl.sum(axis=0)
    de = de_sl.sum(axis=0)
    sc = sc_sl.sum(axis=0)

    
    # 在同一个图层上叠加两张图的结果并保存
    if isOneView == False:
        plt.plot(pr - de, label="prompts-delayed")
        # plt.plot(sc, label="scatter")
    else:
        plt.plot(pr_sl[__view,:] - de_sl[__view,:], label="prompts-delayed")
        # plt.plot(sc_sl[__view,:], label="scatter")

    # plt.ylim([0,2000000])
    plt.legend(fontsize=12)
    
    save_path = __michPath + '/prompt-delayed.png'
    if os.path.exists(save_path):
        os.remove(save_path)  # 删除已存在的文件
    plt.savefig(save_path)  # 保存叠加图形
    # plt.show()
    plt.clf()  # 清除当前图形

    # 在另一个图层上同时画出 prompt, delay的总和并保存
    if isOneView == False:
        plt.plot(pr, label="prompt")
        plt.plot(de, label="delay")
    else:
        plt.plot(pr_sl[__view,:], label="prompt")
        plt.plot(de_sl[__view,:], label="delay")

    # plt.ylim([0,2e6])
    plt.legend(fontsize=12)
    if isOneView == False:
        plt.title("All Views")
        save_path = __michPath + '/prompt&delay_all.png'
        if os.path.exists(save_path):
            os.remove(save_path)
        plt.savefig(__michPath + '/prompt&delay_all.png')  # 保存图形
    else:
        plt.title(f"View {__view}")
        save_path = __michPath + f'/prompt&delay_view_{__view}.png'
        if os.path.exists(save_path):
            os.remove(save_path)
        plt.savefig(__michPath + f'/prompt&delay_view_{__view}.png')  # 保存图形

    plt.show()
    plt.clf()  # 清除当前图形

# for i in range(4890, 4892 + 1):
#     mich_path = f"./Slice{i}"
#     plotFigure(mich_path)
#     for view in range (20, 80 + 1, 30):
#         plotFigure(mich_path, view)
for view in range (20, 80 + 1, 30):
    plotFigure("/home/ustc/Desktop/E180_20250826_WellCounter_PET4643_Slice4796/profile", view)

