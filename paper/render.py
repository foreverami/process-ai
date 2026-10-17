"""llm_algorithm.html을 논문용 PNG(300dpi)와 PDF로 뽑는다.  사용법: python3 paper/render.py"""
import subprocess
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HTML = (HERE / "llm_algorithm.html").as_uri()
COMMON = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--virtual-time-budget=10000"]


def trim(path: Path, pad: int = 36) -> tuple[int, int]:
    img = Image.open(path).convert("RGB")
    img = img.crop(ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox())
    out = Image.new("RGB", (img.width + 2 * pad, img.height + 2 * pad), "white")
    out.paste(img, (pad, pad))
    out.save(path, dpi=(300, 300))
    return out.size


for name, height in (("alg1", 1100), ("alg2", 500), ("notation", 900)):
    png = HERE / f"{name}.png"
    subprocess.run(COMMON + ["--force-device-scale-factor=3", f"--window-size=840,{height}",
                             f"--screenshot={png}", f"{HTML}#{name}"], check=True, capture_output=True)
    print(png.name, trim(png))

subprocess.run(COMMON + ["--no-pdf-header-footer", f"--print-to-pdf={HERE / 'llm_algorithm.pdf'}", HTML],
               check=True, capture_output=True)
print("llm_algorithm.pdf")
