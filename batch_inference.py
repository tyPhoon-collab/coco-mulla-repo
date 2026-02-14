#!/usr/bin/env python
"""
coco-mulla バッチ推論スクリプト

musdb18hqの.labファイルと対応する音声を読み込み、
chord-onlyモードで音声を生成するスクリプト。

Usage:
    python batch_inference.py \
        --chord-dir /app/data/musdb_train \
        --audio-dir /app/data/mixtures \
        --prompt-csv /app/data/description.csv \
        --output-dir /app/out/coco_mulla/generated \
        --model-path ckpt/diff_9_end_0.2.pth \
        --num-samples 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from coco_mulla.models import CoCoMulla
from coco_mulla.utilities import get_device, np2torch
from coco_mulla.utilities.encodec_utils import extract_rvq, save_rvq
from coco_mulla.utilities.sep_utils import separate
from coco_mulla.utilities.symbolic_utils import process_chord
from config import TrainCfg
from tqdm import tqdm

device = get_device()

# Constants
SAMPLE_RATE = TrainCfg.sample_rate  # 32000
FRAME_RES = TrainCfg.frame_res  # 50
# coco-mullaモデルは最大20秒までサポート（max_n_frames=1000）
# TrainCfg.sample_secは47.55だが、coco-mullaでは20秒に制限
SAMPLE_SEC = 20.0
NUM_LAYERS = 48
LATENT_DIM = 12


def load_prompts(csv_path: str | Path) -> dict[str, str]:
    """description.csvからプロンプトを読み込む

    Args:
        csv_path: CSVファイルのパス (key, description, genre)

    Returns:
        トラック名をキー、説明文を値とする辞書
    """
    prompts = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["key"]
            desc = row.get("description", "")
            prompts[key] = desc
    return prompts


def load_chord_only_data(
    audio_path: str | Path,
    chord_path: str | Path,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """chord-onlyモード用のデータを読み込み

    Args:
        audio_path: 音声ファイルパス
        chord_path: 和音ラベルパス
        offset: オフセット秒

    Returns:
        (drums_rvq, midi, chord) のタプル（midiとdrumsはダミー）
    """
    sr = SAMPLE_RATE
    res = FRAME_RES
    sample_sec = SAMPLE_SEC

    # 音声読み込み（ドラム分離用）
    wav, _ = librosa.load(audio_path, sr=sr, mono=True)
    wav = np2torch(wav).to(device)[None, None, ...]

    # ドラム分離
    wavs = separate(wav, sr)
    drums_rvq = extract_rvq(wavs["drums"], sr=sr)

    # 和音処理
    chord, _ = process_chord(chord_path)
    chord_cropped = crop_data(chord[None, ...], "chord", sample_sec, res)

    # N (no chord) のパディングを追加
    if isinstance(chord_cropped, np.ndarray):
        pad_chord = chord_cropped.sum(-1, keepdims=True) == 0
        chord_cropped = np.concatenate([chord_cropped, pad_chord], -1)
    else:
        pad_chord = chord_cropped.sum(-1, keepdim=True) == 0
        chord_cropped = torch.cat([chord_cropped, pad_chord], -1)

    # MIDI は chord-only モードでは使用しないのでゼロパディング
    midi_len = int(sample_sec * res) + 1
    midi = np.zeros((1, midi_len, 128), dtype=np.float32)

    # ドラムRVQをクロップ（drums_rvqはすでにTensor）
    drums_rvq_expanded = drums_rvq[None, ...]  # Add batch dim
    drums_rvq_cropped = crop_data(
        drums_rvq_expanded, "drums_rvq", sample_sec, res, offset=offset
    )

    # テンソル変換
    if isinstance(chord_cropped, np.ndarray):
        chord_tensor = torch.from_numpy(chord_cropped).to(device).float()
    else:
        chord_tensor = chord_cropped.to(device).float()

    midi_tensor = torch.from_numpy(midi).to(device).float()

    if isinstance(drums_rvq_cropped, torch.Tensor):
        drums_tensor = drums_rvq_cropped.to(device).long()
    else:
        drums_tensor = torch.from_numpy(drums_rvq_cropped).to(device).long()

    return drums_tensor, midi_tensor, chord_tensor


def generate_chord_only_mask(xlen: int) -> torch.Tensor:
    """chord-onlyモードのマスクを生成

    Args:
        xlen: シーケンス長

    Returns:
        (1, 2, xlen) のマスクテンソル
    """
    mask = torch.zeros([1, 2, xlen]).to(device)
    # chord-only: drums=0, midi=0
    return mask


def wrap_batch(drums_rvq, midi, chord, cond_mask, prompt):
    """inference.pyのwrap_batch関数を使用（単一サンプル用）

    Args:
        drums_rvq: ドラムRVQ (1, 4, T)
        midi: MIDI (1, T, 128)
        chord: 和音 (1, T, 171)
        cond_mask: 条件マスク (1, 2, T)
        prompt: テキストプロンプト

    Returns:
        バッチ辞書
    """
    batch = {
        "seq": None,
        "desc": [prompt],
        "chords": chord,
        "num_samples": 1,
        "cond_mask": cond_mask,
        "drums": drums_rvq,
        "piano_roll": midi,
        "mode": "inference",
    }
    return batch


def crop(x, mode, sample_sec, res, offset=0):
    """inference.pyのcrop関数をそのまま使用"""
    xlen = x.shape[1] if mode == "chord" or mode == "midi" else x.shape[-1]
    sample_len = int(sample_sec * res) + 1
    if xlen < sample_len:
        if mode == "chord" or mode == "midi":
            x = np.pad(x, ((0, 0), (0, sample_len - xlen), (0, 0)))
        else:
            x = F.pad(x, (0, sample_len - xlen), "constant", 0)
        return x

    st = offset * res
    ed = int((offset + sample_sec) * res) + 1
    if mode == "chord" or mode == "midi":
        assert x.shape[1] > st
        return x[:, st:ed]
    assert x.shape[2] > ed
    return x[:, :, st:ed]


def load_prepared_data(
    input_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
    """準備済みデータを読み込み（ラベルファイルから）

    Args:
        input_dir: サンプルディレクトリ

    Returns:
        (drums_rvq, midi, chord, prompt, track_name)
    """
    sr = SAMPLE_RATE
    res = FRAME_RES
    sample_sec = SAMPLE_SEC

    # メタデータ読み込み
    with open(input_dir / "input.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    prompt = metadata.get("prompt", "")
    track_name = input_dir.name
    lab_path = metadata.get("lab_path")
    start_s = metadata.get("start_s", 0)
    offset = int(start_s)

    if not lab_path or not Path(lab_path).exists():
        raise FileNotFoundError(f"Lab file not found: {lab_path}")

    # 和音処理 (offsetを適用してcrop)
    chord, _ = process_chord(lab_path)
    chord = crop(chord[None, ...], "chord", sample_sec, res, offset=offset)
    pad_chord = chord.sum(-1, keepdims=True) == 0
    chord = np.concatenate([chord, pad_chord], -1)
    chord = torch.from_numpy(chord).to(device).float()

    # 時間次元の長さを和音から取得
    time_len = chord.shape[1]

    # MIDI (ダミー - ゼロ、和音と同じ長さ、offsetあり)
    # inference.pyではoffsetありでcropしているが、ダミーなのでoffsetは不要
    midi = np.zeros((1, time_len, 128), dtype=np.float32)
    midi = torch.from_numpy(midi).to(device).float()

    # ドラム (ダミー - ゼロ、4つのcodebook、和音と同じ長さ)
    drums_rvq = torch.zeros((1, 4, time_len), dtype=torch.long, device=device)

    return drums_rvq, midi, chord, prompt, track_name


def batch_inference(
    chord_dir: str | Path,
    audio_dir: str | Path,
    prompt_csv: str | Path,
    output_dir: str | Path,
    model_path: str | Path,
    num_samples: int | None = None,
    output_sample_rate: int = 44100,
    batch_size: int = 4,
    prepared_input_dir: str | Path | None = None,  # 追加
) -> None:
    """バッチ推論を実行

    Args:
        chord_dir: 和音ラベルディレクトリ
        audio_dir: 音声ディレクトリ
        prompt_csv: プロンプトCSV
        output_dir: 出力ディレクトリ
        model_path: モデルパス
        num_samples: 生成サンプル数 (None または負の値 = 全件)
        output_sample_rate: 出力サンプルレート
        batch_size: 一度に生成するサンプル数
        prepared_input_dir: 準備済み入力ディレクトリ (指定された場合、他の入力引数は無視)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # num_samplesが負の値の場合はNone（全件処理）として扱う
    if num_samples is not None and num_samples < 0:
        num_samples = None

    # モデルロード
    print("Loading model...")
    model = CoCoMulla(SAMPLE_SEC, num_layers=NUM_LAYERS, latent_dim=LATENT_DIM).to(
        device
    )
    model.load_weights(str(model_path))
    model.eval()

    if prepared_input_dir:
        # 準備済みデータを使用
        prepared_input_dir = Path(prepared_input_dir)
        sample_dirs = sorted([d for d in prepared_input_dir.iterdir() if d.is_dir()])
        if num_samples is not None:
            sample_dirs = sample_dirs[:num_samples]

        print(f"Found {len(sample_dirs)} prepared samples in {prepared_input_dir}")

        # シンプルな単一サンプル処理
        for sample_dir in tqdm(sample_dirs, desc="Generating"):
            try:
                drums_rvq, midi, chord, prompt, track_name = load_prepared_data(
                    sample_dir
                )

                # chord-onlyマスク生成（時間次元はdrums_rvqから取得）
                cond_mask = generate_chord_only_mask(drums_rvq.shape[-1])

                # バッチ作成
                batch = wrap_batch(drums_rvq, midi, chord, cond_mask, prompt)

                # 生成
                with torch.no_grad():
                    gen_tokens = model(**batch)

                # 保存
                output_path = output_dir / track_name
                save_rvq(output_list=[str(output_path)], tokens=gen_tokens)

                # メタデータ保存
                metadata = {
                    "track_name": track_name,
                    "prompt": prompt,
                    "model_path": str(model_path),
                    "sample_rate": SAMPLE_RATE,
                    "duration": SAMPLE_SEC,
                    "mode": "chord-only",
                }
                metadata_path = output_dir / f"{track_name}.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"Error processing {sample_dir}: {e}")
                import traceback

                traceback.print_exc()
                continue

    else:
        # 既存のロジック (labファイルから)
        chord_dir = Path(chord_dir)
        audio_dir = Path(audio_dir)

        # プロンプト読み込み
        prompts = load_prompts(prompt_csv)

        # 和音ファイル一覧
        lab_files = sorted(chord_dir.glob("*.lab"))
        if num_samples is not None:
            lab_files = lab_files[:num_samples]

        print(f"Found {len(lab_files)} chord files")

        # 推論ループ
        for lab_file in tqdm(lab_files, desc="Generating"):
            track_name = lab_file.stem

            # 対応する音声ファイルを探す
            audio_file = None
            for ext in [".mp3", ".wav", ".flac"]:
                candidate = audio_dir / f"{track_name}{ext}"
                if candidate.exists():
                    audio_file = candidate
                    break

            if audio_file is None:
                print(f"Warning: No audio file found for {track_name}, skipping...")
                continue

            # プロンプト取得
            prompt = prompts.get(
                track_name, "A musical piece with various instruments."
            )

            try:
                # データロード
                drums_rvq, midi, chord = load_chord_only_data(
                    audio_path=audio_file,
                    chord_path=lab_file,
                    offset=0,
                )

                # chord-onlyマスク生成
                cond_mask = generate_chord_only_mask(drums_rvq.shape[-1])

                # バッチ作成
                batch = wrap_batch(drums_rvq, midi, chord, cond_mask, prompt)

                # 生成
                with torch.no_grad():
                    gen_tokens = model(**batch)

                # 保存
                output_path = output_dir / track_name
                save_rvq(output_list=[str(output_path)], tokens=gen_tokens)

                # メタデータ保存
                metadata = {
                    "track_name": track_name,
                    "prompt": prompt,
                    "chord_path": str(lab_file),
                    "audio_path": str(audio_file),
                    "model_path": str(model_path),
                    "sample_rate": SAMPLE_RATE,
                    "duration": SAMPLE_SEC,
                    "mode": "chord-only",
                }
                metadata_path = output_dir / f"{track_name}.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"Error preparing {track_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

    print(f"Generation complete. Output saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="coco-mulla batch inference for chord-only mode",
    )
    parser.add_argument(
        "--chord-dir",
        type=Path,
        default=Path("data/musdb_train"),
        help="Directory containing chord .lab files",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/mixtures"),
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--prompt-csv",
        type=Path,
        default=Path("data/description.csv"),
        help="CSV file with track prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/coco_mulla/generated"),
        help="Output directory for generated audio",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("ckpt/diff_9_end_0.2.pth"),
        help="Path to model weights",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to generate (None = all)",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=44100,
        help="Output sample rate for final audio (will be resampled)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--prepared-input-dir",
        type=Path,
        default=None,
        help="Directory containing prepared inputs (input.json with lab file path)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    batch_inference(
        chord_dir=args.chord_dir,
        audio_dir=args.audio_dir,
        prompt_csv=args.prompt_csv,
        output_dir=args.output_dir,
        model_path=args.model_path,
        num_samples=args.num_samples,
        output_sample_rate=args.output_sample_rate,
        batch_size=args.batch_size,
        prepared_input_dir=args.prepared_input_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
