import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.ocr_processor import OCRProcessor

cases = {
    'set_single': {'zh_sim'},
    'tuple_single': ('zh_sim',),
    'list_nested': [{'zh_sim'}, ('en',)],
    'str_short': 'zh',
    'empty_list': [],
    'none_val': None,
}

for name, val in cases.items():
    try:
        langs = OCRProcessor(languages=val).languages
    except Exception as e:
        langs = f"ERROR: {e}"
    print(f"{name} -> {langs}")
