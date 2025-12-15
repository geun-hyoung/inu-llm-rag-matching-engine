"""
산학 지식 정보 데이터 처리기
"""

import json
from typing import List, Dict
from pathlib import Path


class DataProcessor:
    """수집된 데이터를 처리하고 정제하는 클래스"""
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_data(self, data: List[Dict]) -> List[Dict]:
        """
        데이터를 정제합니다.
        
        Args:
            data: 원본 데이터 리스트
            
        Returns:
            정제된 데이터 리스트
        """
        cleaned_data = []
        
        for item in data:
            # 빈 값 제거, 공백 정리 등
            cleaned_item = {
                k: v.strip() if isinstance(v, str) else v
                for k, v in item.items()
                if v is not None and v != ""
            }
            cleaned_data.append(cleaned_item)
        
        return cleaned_data
    
    def process_data(self, source: str) -> str:
        """
        데이터를 처리하고 저장합니다.
        
        Args:
            source: 데이터 소스 이름
            
        Returns:
            저장된 파일 경로
        """
        input_file = self.input_dir / f"{source}.json"
        
        if not input_file.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = self.clean_data(raw_data)
        
        output_file = self.output_dir / f"{source}_processed.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"처리된 데이터가 저장되었습니다: {output_file}")
        return str(output_file)


if __name__ == "__main__":
    # 예시 사용법
    processor = DataProcessor()
    processor.process_data("sample_faculty")

