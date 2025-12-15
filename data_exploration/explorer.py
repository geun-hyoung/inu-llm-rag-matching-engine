"""
산학 지식 정보 데이터 탐색기
"""

import json
from typing import List, Dict
from pathlib import Path
from collections import Counter


class DataExplorer:
    """데이터를 탐색하고 분석하는 클래스"""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
    
    def load_data(self, source: str) -> List[Dict]:
        """데이터를 불러옵니다."""
        input_file = self.data_dir / f"{source}_processed.json"
        
        if not input_file.exists():
            return []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def explore_data(self, source: str) -> Dict:
        """
        데이터를 탐색하고 통계를 반환합니다.
        
        Args:
            source: 데이터 소스 이름
            
        Returns:
            데이터 통계 딕셔너리
        """
        data = self.load_data(source)
        
        if not data:
            return {}
        
        stats = {
            "total_count": len(data),
            "fields": list(data[0].keys()) if data else [],
        }
        
        # 전문 분야 통계
        if "expertise" in stats["fields"]:
            all_expertise = []
            for item in data:
                if isinstance(item.get("expertise"), list):
                    all_expertise.extend(item["expertise"])
            stats["expertise_distribution"] = dict(Counter(all_expertise))
        
        # 학과 통계
        if "department" in stats["fields"]:
            departments = [item.get("department") for item in data if item.get("department")]
            stats["department_distribution"] = dict(Counter(departments))
        
        return stats
    
    def save_exploration_results(self, source: str, output_dir: str = "results") -> str:
        """
        탐색 결과를 저장합니다.
        
        Args:
            source: 데이터 소스 이름
            output_dir: 결과 저장 디렉토리
            
        Returns:
            저장된 파일 경로
        """
        results = self.explore_data(source)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"{source}_exploration.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"탐색 결과가 저장되었습니다: {output_file}")
        return str(output_file)


if __name__ == "__main__":
    # 예시 사용법
    explorer = DataExplorer()
    explorer.save_exploration_results("sample_faculty")

