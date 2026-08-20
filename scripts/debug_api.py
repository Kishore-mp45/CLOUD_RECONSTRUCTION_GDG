import sys
from pathlib import Path
sys.path.append(str(Path(".").resolve()))

from api.db.session import SessionLocal
from api.db.models import Result, Scene
from api.routes.results import get_result_preview

def main():
    db = SessionLocal()
    result_id = "inf_20260820_074016_8c136b"
    res = db.query(Result).filter(Result.result_id == result_id).first()
    
    print(f"Result: {res}")
    print(f"job_id: {res.job_id}")
    print(f"preview_png_path: {res.preview_png_path}")
    print(f"scene: {res.scene}")
    
    out_recon_png = Path(res.preview_png_path).parent / f"{res.job_id}_reconstructed_rgb_v7.png"
    print(f"out_recon_png path: {out_recon_png}")
    
    if res.scene and res.scene.s2_path:
        print(f"s2_path: {res.scene.s2_path}")
    
    try:
        resp = get_result_preview(result_id=result_id, modality="reconstructed", db=db)
        print(f"Response: {resp}")
        print(f"Path in response: {resp.path}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()
