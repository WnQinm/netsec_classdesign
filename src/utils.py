from typing import Dict, List, Union, Optional
from collections import defaultdict, Counter
import logging
import random
import json
import os

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from pyshark.packet.packet import Packet
import pyshark
from sklearn.metrics import recall_score, accuracy_score, f1_score


TSHARK_PATH = "D:\\Scoop\\apps\\wireshark\\current\\tshark.exe"


# 层类型:
# ipv6/ip - tcp/udf - other layer
# arp
def get_data_from_cap_file(
    cap_path: Union[str, List[str]],
    save_path: Optional[str] = None,
    cap_len: Optional[List[int]] = None
) -> Optional[Dict[str, List[Packet]]]:
    '''
    cap_path: file or dir[file] to get data
    save_path: If not empty, the result is saved to the specified directory, otherwise the result is returned
    '''
    if os.path.isdir(cap_path):
        cap_paths = [os.path.join(cap_path, file) for file in os.listdir(cap_path)]
        file_pbar = tqdm(total=len(cap_paths), position=1)
        if save_path:
            with logging_redirect_tqdm():
                logging.info(f"all data from {cap_path} will be saved in {save_path}")
    elif os.path.isfile(cap_path):
        cap_paths = [cap_path]
        file_pbar = None
    else:
        raise Exception("path must lead to a file or directory")
    if cap_len:
        assert len(cap_len) == len(cap_paths)

    # 依据五元组提取流  流内部按时间戳排序
    results = defaultdict(list)
    for path in cap_paths:
        if cap_len:
            cap_pbar = tqdm(total=cap_len[cap_paths.index(path)], position=2)
            cap_pbar.set_description(path)

        caps = pyshark.FileCapture(path, tshark_path=TSHARK_PATH)
        for cap in caps:
            layers = [x._layer_name for x in cap.layers]
            result = dict()
            content = [str(layers)]

            # frame info
            info = cap.frame_info
            content.append(f"frame info: Capture Length {info.cap_len}, Frame Length {info.len}, Frame Number {info.number}, Time since reference or first frame {info.time_relative}")

            # ip info
            if 'ip' in layers:
                result["src"] = cap.ip.src
                result["dst"] = cap.ip.dst
                ip = cap.ip
                content.append(f"{ip._layer_name}: Source Address {ip.src}, Destination Address {ip.dst}, Total Length {ip.len}, Stream index {ip.stream}, Time to Live {ip.ttl}")
            elif 'ipv6' in layers:
                result["src"] = cap.ipv6.src
                result["dst"] = cap.ipv6.dst
                ip = cap.ipv6
                content.append(f"{ip._layer_name}: Source Address {ip.src}, Destination Address {ip.dst}, Payload Length {ip.plen}, Stream index {ip.stream}")
            else:
                if cap_len:
                    cap_pbar.update(1)
                    if cap_pbar.n > cap_pbar.total:
                        break
                continue

            # tcp/udp info
            if 'tcp' in layers:
                result["protocol"] = "tcp"
                result["src"] += (":"+cap.tcp.port)
                result["dst"] += (":"+cap.tcp.dstport)
                content.append(f"{cap.tcp._layer_name}: Source Port {cap.tcp.port}, Destination Port {cap.tcp.dstport}, TCP Segment Len {cap.tcp.len}, Sequence Number {cap.tcp.seq}, Next Sequence Number {cap.tcp.nxtseq}, Time since previous frame in this TCP stream {cap.tcp.time_delta}, Time since first frame in this TCP stream {cap.tcp.time_relative}, Window {cap.tcp.window_size}")
            elif 'udp' in layers:
                result["protocol"] = "udp"
                result["src"] += (":"+cap.udp.port)
                result["dst"] += (":"+cap.udp.dstport)
                content.append(f"{cap.udp._layer_name}: Source Port {cap.udp.port}, Destination Port {cap.udp.dstport}, Length {cap.udp.length}, Stream Packet Number {cap.udp.stream_pnum}, Time since previous frame {cap.udp.time_delta}, Time since first frame {cap.udp.time_relative}")
            else:
                # raise NotImplementedError(f"cannot get port data from {layers} in file {path}")
                if cap_len:
                    cap_pbar.update(1)
                    if cap_pbar.n > cap_pbar.total:
                        break
                continue

            results[f'{result["protocol"]}-{"-".join(sorted([result["src"],result["dst"]]))}'].append([info.time, "; ".join(content)])
            if cap_len:
                cap_pbar.update(1)
                if cap_pbar.n > cap_pbar.total:
                    break
        caps.close()

        if file_pbar:
            file_pbar.update(1)

    results = {k:[res[1] for res in sorted(v, key=lambda x: x[0])] for k,v in results.items()}

    if save_path:
        import json
        with open(save_path, "w") as f:
            f.write(json.dumps(results))
        with logging_redirect_tqdm():
            logging.info(f"save processed cap data into {save_path}")
    else:
        return results


# [{"label": xxx, "data": [xxx, ...]}, ...]
def construct_dataset(
    fold_path: str,
    patch_size: int = 10,
    data_size: int = 1024,
    save_path: Optional[str] = None,
) -> Optional[List[Dict[str, str]]]:
    '''
    fold_path: 数据集路径
    patch_size: 预测时要用到的流数目
    data_size: 每个label的数据量
    save_path: 保存路径
    '''
    assert os.path.isdir(fold_path)
    result = []
    for file in os.listdir(fold_path):
        label = file.split(".")[0]
        with open(os.path.join(fold_path, file), "r") as f:
            raw_data:Dict[str, List[str]] = json.load(f)

        prob = []
        idxs = []
        for v in raw_data.values():
            n = max(len(v) - patch_size, 1)
            prob.append(n)
            idxs.append(range(n))
        # 根据可选取的子流数量确定每个流被选取的数量
        prob = list(map(lambda x: x / sum(prob), prob))
        select_size = Counter(random.choices(range(len(prob)), weights=prob, k=data_size))
        select_size.update({i: 0 for i in range(len(prob))})
        _, select_size = zip(*sorted(list(select_size.items()), key=lambda x: x[0]))
        # 按照select_size在raw_data.values()中随机选取长度最长为patch_size的流
        selected_data = [random.choices(population=p, k=k) for p,k in zip(idxs, select_size)]
        selected_data = sum(
            [
                [d[i : i + patch_size] if (i + patch_size) <= len(d) else d[i:] for i in ids]
                for ids, d in zip(selected_data, raw_data.values()) if len(d) > 0
            ],
            [],
        )
        result.extend([{'label': label, 'data': d} for d in selected_data])

    if save_path:
        with open(save_path, "w") as f:
            f.write(json.dumps(result))
    else:
        return result


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    recall = recall_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {
        "accuracy": acc,
        "recall": recall,
        "f1": f1
    }