# cobra/conf/pct.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Sequence
import os
import json


DEFAULT_TARGETS: List[str] = ["vision.siglip", "vision.dino", "llm", "projector"]
ROTATE_DEFAULT: Dict[str, Any] = {
    "rotation": "hadamard",
    "axis": "io",
    "scope": "per_channel",
    "block_size": 64,
    "method": "bake",
    "klt_batches": 8,
    "eps": 1e-6,
}
FINALIZE_DEFAULT: Dict[str, Any] = {
    "weight_bits": 8,
    "act_bits": 8,
    "symmetric": True,
    "per_channel": True,
    "rounding": "nearest",
    "pack_linear": True,
    "pack_conv": True,
    "verify_batches": 0,
    "atol": 1e-2,
    "rtol": 1e-1,
}


@dataclass
class PctConfig:
    """
    Percentile Clipping (PCT) configuration.

    Fields
    ------
    enabled : 是否啟用整個百分位裁剪觀測/套用管線
    targets : 四個固定模塊鍵名；可自定子集
    percentile_override : 若設置，四模塊一律採用該百分位；否則走 best-percentile 規則
    stats_out : 蒐集階段輸出 .pt 路徑（accumulator 狀態）
    overrides_json : pct_apply 產生的裁剪表（lo/hi/percentile）
    device : 收集與推論的預設裝置
    img_size : 影像短邊縮放與中心裁切尺寸（僅供收集腳本使用）
    text_len : 文字序列長度（隨機 token，用於驅動 llm embedding；僅供收集腳本使用）
    batches : 收集批次數
    batch_size : 每批大小
    """
    enabled: bool = False
    targets: List[str] = field(default_factory=lambda: list(DEFAULT_TARGETS))
    percentile_override: Optional[float] = None

    stats_out: str = "outputs/percentile_stats.pt"
    overrides_json: str = "outputs/percentile_overrides.json"

    device: str = "cuda"
    img_size: int = 384
    text_len: int = 16
    batches: int = 8
    batch_size: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PctConfig":
        cfg = cls()
        for k, v in (obj or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        # 型別與內容保護
        if not isinstance(cfg.targets, list) or not cfg.targets:
            cfg.targets = list(DEFAULT_TARGETS)
        return cfg

    @classmethod
    def from_env(cls, prefix: str = "PCT_") -> "PctConfig":
        """
        以環境變數覆寫常用欄位。
        支援：
          PCT_ENABLED=0|1
          PCT_TARGETS=vision.siglip,vision.dino,llm,projector
          PCT_PERCENTILE_OVERRIDE=99.99
          PCT_STATS_OUT=...
          PCT_OVERRIDES_JSON=...
          PCT_DEVICE=cuda|cpu
          PCT_IMG_SIZE=384
          PCT_TEXT_LEN=16
          PCT_BATCHES=8
          PCT_BATCH_SIZE=4
        """
        cfg = cls()
        def _get(name: str, default: Optional[str] = None) -> Optional[str]:
            return os.environ.get(prefix + name, default)

        if (v := _get("ENABLED")) is not None:
            cfg.enabled = v.strip() not in ("0", "false", "False", "")

        if (v := _get("TARGETS")):
            targets = [t.strip() for t in v.split(",") if t.strip()]
            if targets:
                cfg.targets = targets

        if (v := _get("PERCENTILE_OVERRIDE")):
            try:
                cfg.percentile_override = float(v)
            except ValueError:
                cfg.percentile_override = None

        if (v := _get("STATS_OUT")):
            cfg.stats_out = v
        if (v := _get("OVERRIDES_JSON")):
            cfg.overrides_json = v
        if (v := _get("DEVICE")):
            cfg.device = v

        if (v := _get("IMG_SIZE")):
            try: cfg.img_size = int(v)
            except ValueError: pass
        if (v := _get("TEXT_LEN")):
            try: cfg.text_len = int(v)
            except ValueError: pass
        if (v := _get("BATCHES")):
            try: cfg.batches = int(v)
            except ValueError: pass
        if (v := _get("BATCH_SIZE")):
            try: cfg.batch_size = int(v)
            except ValueError: pass

        return cfg

    def merge(self, other: Optional[Dict[str, Any]] = None) -> "PctConfig":
        """
        以 dict 覆寫當前設定，回傳 self。
        """
        if not other:
            return self
        for k, v in other.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if not isinstance(self.targets, list) or not self.targets:
            self.targets = list(DEFAULT_TARGETS)
        return self

    # 便捷 I/O

    @classmethod
    def load_json(cls, path: str) -> "PctConfig":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls.from_dict(obj)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


__all__ = ["PctConfig", "DEFAULT_TARGETS", "ROTATE_DEFAULT", "FINALIZE_DEFAULT"]
