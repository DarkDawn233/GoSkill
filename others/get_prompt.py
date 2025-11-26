from pathlib import Path
import numpy as np

data_path = Path("/data1/metaworld")

prompt_dict = {
    "assembly-v2": [(737, 820, 919, 928), (1092, 1416, 1359, 1896)],
    "basketball-v2": [(826, 513, 855, 615), (1229, 1267, 1685, 1257)],
    "bin-picking-v2": [(952, 999, 959, 908), (1332, 1359, 1968, 1312)],
    "box-close-v2": [(974, 531, 519, 914), (1868, 1758, 1627, 974)],
    "button-press-topdown-v2": [(913, 918, 901, 998), (1474, 1984, 1917, 1953)],
    "button-press-topdown-wall-v2": [(739, 983, 566, 550), (1876, 1840, 1827, 1852)],
    "button-press-v2": [(523, 866, 581, 892), (1283, 1835, 1162, 1036)],
    "button-press-wall-v2": [(333, 933, 326, 926), (1757, 1776, 1930, 1854)],
    "coffee-button-v2": [(592, 265, 865, 438), (1861, 1716, 1811, 1813)],
    "coffee-pull-v2": [(444, 647, 800, 518), (1225, 1308, 1928, 1948)],
    "coffee-push-v2": [(152, 601, 966, 657), (152, 1653, 1508, 1834)],
    "dial-turn-v2": [(924, 829, 859, 991), (1781, 1999, 1804, 1647)],
    "disassemble-v2": [(717, 842, 650, 627), (1489, 1473, 1765, 1871)],
    "door-close-v2": [(472, 168, 768, 333), (1695, 472, 1072, 1634)],
    "door-lock-v2": [(147, 594, 994, 289), (1949, 1626, 1667, 1930)],
    "door-open-v2": [(562, 962, 576, 976), (1739, 1938, 1156, 1356)],
    "door-unlock-v2": [(312, 712, 529, 929), (1173, 1373, 1400, 1730)],
    "drawer-close-v2": [(576, 543, 717, 406), (576, 1593, 543, 1125)],
    "drawer-open-v2": [(999, 828, 926, 911), (1046, 1740, 1481, 1047)],
    "faucet-close-v2": [(654, 848, 750, 720), (1065, 1770, 1090, 1136)],
    "faucet-open-v2": [(406, 806, 274, 674), (1131, 1331, 406, 806)],
    "hammer-v2": [(729, 948, 911, 629), (1875, 1993, 1982, 1567)],
    "hand-insert-v2": [(724, 987, 906, 792), (1824, 1282, 1864, 1808)],
    "handle-press-side-v2": [(371, 971, 378, 978), (1827, 1660, 1786, 1912)],
    "handle-press-v2": [(235, 716, 155, 567), (235, 716, 1076, 155)],
    "handle-pull-side-v2": [(578, 429, 501, 487), (1947, 1580, 1599, 1763)],
    "handle-pull-v2": [(197, 797, 308, 908), (1674, 197, 797, 1830)],
    "lever-pull-v2": [(909, 978, 910, 877), (1518, 1850, 1544, 1731)],
    "peg-insert-side-v2": [(904, 972, 807, 870), (1554, 1557, 1979, 1961)],
    "peg-unplug-side-v2": [(992, 981, 885, 826), (1825, 1875, 1862, 1725)],
    "pick-out-of-hole-v2": [(986, 894, 888, 802), (1811, 1866, 1889, 1154)],
    "pick-place-v2": [(622, 495, 39, 674), (1921, 1677, 1070, 1488)],
    "pick-place-wall-v2": [(879, 512, 800, 957), (1737, 1785, 1901, 1649)],
    "plate-slide-back-side-v2": [(110, 710, 147, 747), (110, 710, 147, 747)],
    "plate-slide-back-v2": [(475, 450, 321, 921), (1785, 1787, 475, 1075)],
    "plate-slide-side-v2": [(502, 508, 509, 498), (1774, 502, 1102, 1386)],
    "plate-slide-v2": [(353, 640, 978, 970), (1927, 1874, 353, 1386)],
    "push-back-v2": [(889, 689, 696, 608), (1427, 1075, 1314, 1970)],
    "push-v2": [(744, 975, 757, 949), (1894, 1802, 1979, 744)],
    "push-wall-v2": [(753, 617, 882, 896), (1738, 1747, 1862, 1957)],
    "reach-v2": [(788, 533, 821, 71), (1806, 1531, 1471, 1643)],
    "reach-wall-v2": [(860, 984, 345, 975), (1919, 1611, 1887, 1980)],
    "shelf-place-v2": [(840, 866, 896, 760), (1872, 1654, 1792, 1888)],
    "soccer-v2": [(869, 870, 998, 780), (1956, 1790, 1769, 1333)],
    "stick-pull-v2": [(821, 718, 789, 953), (1898, 1950, 821, 1258)],
    "stick-push-v2": [(900, 964, 895, 988), (1483, 1490, 1539, 1295)], 
    "sweep-into-v2": [(930, 654, 433, 553), (1623, 1995, 930, 654)],
    "sweep-v2": [(968, 948, 926, 900), (1623, 1174, 1974, 1582)],
    "window-close-v2": [(510, 571, 349, 949), (1592, 1866, 1985, 1990)],
    "window-open-v2": [(599, 534, 388, 988), (1861, 1913, 1948, 1971)]
}

def check_data():
    for env_dir in data_path.iterdir():
        if "-v2" not in str(env_dir):
            continue
        print(env_dir)
        returns = []
        obss = []
        for i in range(2000):
            data_file = env_dir / f"{i}.npz"
            data = np.load(data_file)
            observations = data["observations"]
            actions = data["actions"]
            rewards = data["rewards"]
            next_observations = data["next_observations"]
            terminals = data["terminals"]
            task = data["task"]
            if observations.shape[0] != 500 or actions.shape[0] != 500 or rewards.shape[0] != 500 or next_observations.shape[0] != 500 or terminals.shape[0] != 500:
                print("Error data:", data_file)
                print(observations.shape, actions.shape, rewards.shape, next_observations.shape, terminals.shape, task.shape)
            returns.append((i, np.sum(rewards)))
            if i == 999:
                returns = sorted(returns, key=lambda x: x[1], reverse=True)
                id = [returns[_][0] for _ in range(4)]
                print(id)
                for j in range(4):
                    print(returns[j])
        print("")
        returns = sorted(returns, key=lambda x: x[1], reverse=True)
        id = [returns[_][0] for _ in range(4)]
        print(id)
        for j in range(4):
            print(returns[j])

def gen_prompt():
    for env_name, prompt_list in prompt_dict.items():
        env_dir = data_path / env_name
        prompt_dir = data_path / "prompt"
        prompt_dir.mkdir(exist_ok=True)
        for prompt_mode, data_ids in zip(["medium", "expert"], prompt_list):
            file_path = prompt_dir / f"{env_name}-prompt-{prompt_mode}.npy"
            data = []
            for i in data_ids:
                data_file = env_dir / f"{i}.npz"
                data_i = np.load(data_file)
                data_i = {k: data_i[k] for k in data_i.keys()}
                data.append(data_i)
            
            with open(file_path, "wb") as f:
                np.save(f, data)

gen_prompt()



        