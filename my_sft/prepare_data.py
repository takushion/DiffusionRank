import json
import gzip
import os
import re
import urllib.request


RANK_ZEPHYR_URL = (
    "https://huggingface.co/datasets/castorini/"
    "rank_zephyr_training_data/resolve/main/"
    "rank_zephyr_training_data.jsonl.gz"
)
OUTPUT_PATH = "datasets/rank_gpt4_all.jsonl"


def download_rankzephyr(url: str) -> str:
    """RankZephyr データセットをダウンロードし、展開した文字列を返す"""
    print(f"Downloading from {url}...")
    with urllib.request.urlopen(url) as resp:
        compressed = resp.read()
    raw = gzip.decompress(compressed).decode()
    print(f"Downloaded {len(compressed) / 1024 / 1024:.1f} MB "
          f"(uncompressed {len(raw) / 1024 / 1024:.1f} MB)")
    return raw


def parse_conversations(human: str, gpt: str) -> dict:
    """RankZephyr の会話ターンから query, document, ranking を抽出する"""

    # 1. クエリを抽出 "Search Query: <query>" の後ろ
    m = re.search(r"Search Query:\s*(.+?)(?:\n|$)", human)
    if not m:
        raise ValueError("No Search Query found in human message")
    query = m.group(1).strip()

    # 2. 文書を抽出 行頭 [N] の行
    doc_lines = re.findall(r"^\[\d+\]\s*(.*)", human, re.MULTILINE)
    documents = [line.strip() for line in doc_lines]

    # 3. ランキングを抽出 "[N] > [M] > [L] ..." から数値リストに
    ranking = [int(x) for x in re.findall(r"\d+", gpt)]

    return {"query": query, "document": documents, "ranking": ranking}


def convert(raw_data: str) -> list[dict]:
    """生データの全行をパースしてリストに変換"""
    rows = []
    for i, line in enumerate(raw_data.strip().split("\n")):
        item = json.loads(line)
        convs = item["conversations"]
        try:
            row = parse_conversations(
                human=convs[1]["value"],
                gpt=convs[2]["value"],
            )
        except (KeyError, ValueError, IndexError) as e:
            print(f"Skipping row {i} ({item.get('id', '?')}): {e}")
            continue
        rows.append(row)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} / {len(raw_data.strip().split(chr(10)))}...")

    return rows


def save_jsonl(rows: list[dict], path: str):
    """JSONL 形式で保存"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} samples to {path}")


def main():
    raw = download_rankzephyr(RANK_ZEPHYR_URL)
    rows = convert(raw)
    save_jsonl(rows, OUTPUT_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
