"""
산학 지식 정보 데이터 수집기
"""

import json
from typing import List, Dict
from pathlib import Path


class DataCollector:
    """산학 지식 정보를 수집하는 클래스"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_data(self, source: str, data: List[Dict]) -> str:
        """
        데이터를 수집하고 저장합니다.
        
        Args:
            source: 데이터 소스 이름
            data: 수집할 데이터 리스트
            
        Returns:
            저장된 파일 경로
        """
        output_file = self.output_dir / f"{source}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"데이터가 저장되었습니다: {output_file}")
        return str(output_file)
    
    def load_data(self, source: str) -> List[Dict]:
        """
        저장된 데이터를 불러옵니다.
        
        Args:
            source: 데이터 소스 이름
            
        Returns:
            데이터 리스트
        """
        input_file = self.output_dir / f"{source}.json"
        
        if not input_file.exists():
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)


if __name__ == "__main__":
    # 예시 사용법
    collector = DataCollector()
    
    # 샘플 데이터 수집
    sample_data = [
        {
            "name": "홍길동",
            "department": "컴퓨터공학과",
            "expertise": ["인공지능", "머신러닝"],
            "research_areas": ["딥러닝", "자연어처리"]
        }
    ]
    
    collector.collect_data("sample_faculty", sample_data)

