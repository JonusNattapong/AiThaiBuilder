import os
import pandas as pd
import argparse
from typing import Dict, Any, List

class AugmentationInspector:
    def __init__(self, augmented_data: str, text_column: str, original_text_column: str, technique_column: str, label_column: str, output_dir: str):
        self.data = pd.read_csv(augmented_data)
        self.text_column = text_column
        self.original_text_column = original_text_column
        self.technique_column = technique_column
        self.label_column = label_column
        self.output_dir = output_dir

    def get_technique_distribution(self) -> Dict[str, int]:
        return self.data[self.technique_column].value_counts().to_dict()

    def get_text_length_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for technique, group in self.data.groupby(self.technique_column):
            lengths = group[self.text_column].str.len()
            stats[technique] = {
                'mean': lengths.mean(),
                'median': lengths.median(),
                'min': lengths.min(),
                'max': lengths.max(),
                'std': lengths.std()
            }
        return stats

    def plot_technique_distribution(self):
        pass

    def plot_text_length_distribution(self):
        pass

    def plot_similarity_distribution(self):
        pass

    def plot_label_distribution(self):
        pass

    def calculate_similarity_to_original(self) -> Dict[str, Dict[str, Any]]:
        pass

    def find_problematic_augmentations(self, min_similarity: float, max_similarity: float) -> Dict[str, List[Dict[str, Any]]]:
        pass

    def get_label_distribution(self) -> Dict[str, Dict[str, int]]:
        pass

    def generate_report(self, save_report: bool = True) -> Dict[str, Any]:
        """สร้างรายงานสรุปผลการวิเคราะห์"""
        # ...existing code...

        # บันทึกรายงาน
        if save_report:
            import json
            output_path = os.path.join(self.output_dir, 'augmentation_report.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            # สร้างกราฟสำหรับรายงาน
            self.plot_technique_distribution()
            self.plot_text_length_distribution()
            
            if self.original_text_column in self.data.columns:
                self.plot_similarity_distribution()
            
            if self.label_column:
                self.plot_label_distribution()
            
            # สร้าง HTML report สรุป
            self._generate_html_report(report, output_path.replace('.json', '.html'))
        
        return report
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: str) -> None:
        """สร้าง HTML report จากข้อมูลรายงาน"""
        html_content = [
            '<!DOCTYPE html>',
            '<html lang="th">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<title>รายงานการวิเคราะห์ Data Augmentation</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }',
            'h1, h2, h3 { color: #333366; }',
            'table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }',
            'th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }',
            'th { background-color: #f2f2f2; }',
            'tr:nth-child(even) { background-color: #f9f9f9; }',
            'img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #eee; }',
            '.section { margin-bottom: 30px; border: 1px solid #eee; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }',
            '.metric { display: inline-block; margin: 10px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; text-align: center; min-width: 120px; }',
            '.metric .value { font-size: 24px; font-weight: bold; color: #3366cc; }',
            '.metric .label { font-size: 14px; color: #666; }',
            '.warning { background-color: #fff8e1; border-left: 5px solid #ffc107; padding: 10px; margin: 10px 0; }',
            '.success { background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 10px; margin: 10px 0; }',
            '</style>',
            '</head>',
            '<body>',
            '<h1>รายงานการวิเคราะห์ Data Augmentation</h1>'
        ]
        
        # ข้อมูลทั่วไปของชุดข้อมูล
        html_content.extend([
            '<div class="section">',
            '<h2>ข้อมูลชุดข้อมูล</h2>',
            '<div class="metric">',
            f'<div class="value">{report["dataset_info"]["total_samples"]}</div>',
            '<div class="label">จำนวนตัวอย่างทั้งหมด</div>',
            '</div>',
            '<div class="metric">',
            f'<div class="value">{report["dataset_info"]["original_samples"]}</div>',
            '<div class="label">ตัวอย่างต้นฉบับ</div>',
            '</div>',
            '<div class="metric">',
            f'<div class="value">{report["dataset_info"]["augmented_samples"]}</div>',
            '<div class="label">ตัวอย่างที่ augment</div>',
            '</div>'
        ])
        
        # ข้อมูลเทคนิคที่ใช้
        html_content.extend([
            '<h3>การกระจายของเทคนิค</h3>',
            '<table>',
            '<tr><th>เทคนิค</th><th>จำนวนตัวอย่าง</th></tr>'
        ])
        
        for technique, count in report["techniques"]["distribution"].items():
            html_content.append(f'<tr><td>{technique}</td><td>{count}</td></tr>')
        
        html_content.extend([
            '</table>',
            '<img src="technique_distribution.png" alt="การกระจายของเทคนิค">',
            '</div>'
        ])
        
        # ข้อมูลความยาวข้อความ
        html_content.extend([
            '<div class="section">',
            '<h2>ความยาวข้อความ</h2>',
            '<img src="text_length_distribution.png" alt="การกระจายความยาวข้อความ">',
            
            '<h3>สถิติความยาวข้อความตามเทคนิค</h3>',
            '<table>',
            '<tr><th>เทคนิค</th><th>ค่าเฉลี่ย</th><th>ค่ามัธยฐาน</th><th>ค่าต่ำสุด</th><th>ค่าสูงสุด</th><th>ส่วนเบี่ยงเบนมาตรฐาน</th></tr>'
        ])
        
        for technique, stats in report["text_length"]["stats"].items():
            html_content.append(
                f'<tr><td>{technique}</td><td>{stats["mean"]:.2f}</td><td>{stats["median"]:.2f}</td>'