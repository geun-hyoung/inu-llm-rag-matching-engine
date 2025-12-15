"""
결과 저장 유틸리티
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class ResultsSaver:
    """실험 및 분석 결과를 저장하는 클래스"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def save_result(self, data: Any, filename: str, subdir: str = None):
        """
        결과를 저장합니다.
        
        Args:
            data: 저장할 데이터
            filename: 파일명
            subdir: 하위 디렉토리 (선택)
        """
        if subdir:
            output_dir = self.results_dir / subdir
        else:
            output_dir = self.results_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / filename
        
        if isinstance(data, (dict, list)):
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(str(data))
        
        print(f"결과가 저장되었습니다: {output_file}")
        return str(output_file)
    
    def save_with_timestamp(self, data: Any, prefix: str, subdir: str = None):
        """
        타임스탬프를 포함한 파일명으로 결과를 저장합니다.
        
        Args:
            data: 저장할 데이터
            prefix: 파일명 접두사
            subdir: 하위 디렉토리 (선택)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        return self.save_result(data, filename, subdir)


if __name__ == "__main__":
    # 예시 사용법
    saver = ResultsSaver()
    
    sample_result = {
        "experiment_name": "embedding_test",
        "accuracy": 0.95,
        "top_k": 5
    }
    
    saver.save_with_timestamp(sample_result, "experiment_result")

